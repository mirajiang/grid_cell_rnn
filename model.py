import torch
import numpy as np
import matplotlib.pyplot as plt

class RNN(torch.nn.Module):
    def __init__(self, options, place_cells):
        super(RNN, self).__init__()
        self.Ng = options.Ng #Gives number of grid cells
        self.Np = options.Np #Gives number place cells
        self.sequence_length = options.sequence_length 
        self.weight_decay = options.weight_decay
        self.place_cells = place_cells

        # Velocity and hidden streams attenuation and noise
        self.vel_sigma = options.vel_sigma
        self.vel_scale = options.vel_scale
        self.hid_sigma = options.hid_sigma
        self.hid_scale = options.hid_scale

        self.device = options.device

        # Neuron mask
        if options.neuron_lesion_prob is not None:
            self.neuron_mask = torch.full((self.Ng, ), 1.0 - options.neuron_lesion_prob)
            self.neuron_mask = torch.bernoulli(self.neuron_mask).to(self.device)

        # Input weights
        self.encoder = torch.nn.Linear(self.Np, self.Ng, bias=False)

        # RNN
        self.vel_stream = torch.nn.Linear(2, self.Ng, bias=False) #Input is velocity stream (vx, vy)
        self.hid_stream = torch.nn.Linear(self.Ng, self.Ng, bias=False) #Takes hidden layer from previous timestep
        if (options.activation == 'relu'):
            self.activation = torch.nn.ReLU() # Could make sigmoid or ReLU
        elif (options.activation == 'sigmoid'):
            self.activation = torch.nn.Sigmoid()
        else:
            print('Not a valid activation')
            return

        # Linear read-out weights
        self.decoder = torch.nn.Linear(self.Ng, self.Np, bias=False)

        self.softmax = torch.nn.Softmax(dim=-1)

    def grid(self, inputs, store_neurons=False):
        '''
        Compute grid cell activations.
        Args:
            inputs: Tuple of velocity input and the initial state for RNN ([sequence_length, batch_size, 2], [batch_size, Np]).

        Returns: 
            g: Batch of grid cell activations with shape [sequence_length, batch_size, Ng].
        '''

        vt, p0 = inputs # Unpacks velocity vector and place cell activations
        #assert vt.shape[0] == self.sequence_length and vt.shape[2] == 2, vt.shape # Checks vt is of correct shape

        h = self.encoder(p0) # Inputs place cell activations to encode position for the first time step
        g = torch.zeros(vt.shape[0], vt.shape[1], self.Ng, device=self.device)
        if(store_neurons):
            self.neuron_values = np.zeros((vt.shape[0], self.Ng))

        for i, v in enumerate(vt):
            # Add attenuation (scale) or noise (sigma) to the velocity and hidden streams
            vp = self.vel_scale * v + self.vel_sigma * torch.randn(v.shape,device=self.device)
            hp = self.hid_scale * h + self.hid_sigma * torch.randn(h.shape,device=self.device)

            vs = self.vel_stream(vp)
            hs = self.hid_stream(hp)
            h = self.activation(vs + hs)

            # Mask neurons
            if self.neuron_mask is not None:
                h *= self.neuron_mask

            if(store_neurons):
                self.neuron_values[i] = h[0].cpu().detach().numpy()

            g[i] = h

        return g

    def predict(self, inputs):
        '''
        Predict place cell code.
        Args:
            inputs: Tuple of velocity input and the initial state for RNN ([sequence_length, batch_size, 2], [batch_size, Np]).

        Returns: 
            place_preds: Predicted place cell activations with shape 
                [sequence_length, batch_size, Np].
        '''

        place_preds = self.decoder(self.grid(inputs))
        return place_preds
    
    def compute_loss(self, inputs, pc_outputs, pos):
        '''
        Compute avg. loss and decoding error.
        Args:
            inputs: Tuple of velocity input and the initial state for RNN ([sequence_length, batch_size, 2], [batch_size, Np]).
            pc_outputs: Ground truth place cell activations with shape [sequence_length, batch_size, Np].
            pos: Ground truth 2d position with shape [sequence_length, batch_size, 2].

        Returns:
            loss: Avg. loss for this training batch.
            err: Avg. decoded position error in cm.
        '''

        y = pc_outputs
        preds = self.predict(inputs)
        y_hat = self.softmax(preds)
        loss = -(y*torch.log(y_hat)).sum(-1).mean()

        # L2 weight regularization
        loss += self.weight_decay * (self.hid_stream.weight**2).sum()

        # Compute decoding error
        pred_pos = self.place_cells.get_nearest_cell_pos(preds)
        err = torch.sqrt(((pos - pred_pos)**2).sum(-1)).mean()

        return loss, err