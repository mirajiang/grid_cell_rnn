import numpy as np
import torch
import matplotlib.pyplot as plt
import utils

from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator
from model import RNN
from trainer import Trainer


class Options():
    pass
options = Options()

# Model 1
options.save_dir = '/home/mira/Grid_Cell_RNN/models'    # directory to save trained models
options.n_epochs = 100      # number of training epochs
options.n_steps = 1000      # number of training steps (batches per epoch)
options.batch_size = 200      # number of trajectories per batch (originally 200)
options.sequence_length = 50  # number of steps per trajectory (originally 20)
options.activation = 'relu'     # activation function chosen (relu or sigmoid)
options.learning_rate = 1e-4  # gradient descent learning rate
options.Np = 512              # number of place cells
options.Ng = 2500             # number of grid cells
options.place_cell_rf = 0.12  # width of place cell center tuning curve (m)
options.surround_scale = 2    # if DoG, ratio of sigma2^2 to sigma1^2
options.weight_decay = 1e-4   # strength of weight decay on recurrent weights
options.DoG = True            # use difference of gaussians tuning curves
options.periodic = False      # trajectories with periodic boundary conditions
options.box_width = 2.2       # width of training environment
options.box_height = 2.2      # height of training environment
options.vel_sigma = 0       # noise in velocity stream
options.vel_scale = 1.0     # attenuation of velocity stream
options.hid_sigma = 0      # noise in hidden stream
options.hid_scale = 1.0        # attenuation of hidden stream
options.replication_num = '01' # ID to differentiate models trained on the same parameters (optional)

# Generate run ID
options.run_ID = utils.generate_run_ID(options)

# Put onto GPU if available
options.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# Generate model and trajectories
place_cells = PlaceCells(options)

model = RNN(options, place_cells).to(options.device)
trajectory_generator = TrajectoryGenerator(options, place_cells)
trainer = Trainer(options, model, trajectory_generator)

trainer.train(n_epochs=options.n_epochs, n_steps=options.n_steps, save=True)

err_np = np.array([item.cpu().item() if isinstance(item, torch.Tensor) else item for item in trainer.err])
loss_np = np.array([item.cpu().item() if isinstance(item, torch.Tensor) else item for item in trainer.loss])

plt.figure(figsize=(12,3))
plt.subplot(121)
plt.plot(err_np, c='black')

plt.title('Decoding error (m)'); plt.xlabel('train step')
plt.subplot(122)
plt.plot(loss_np, c='black')
plt.title('Loss'); plt.xlabel('train step')

plt.savefig('/home/mira/Grid_Cell_RNN/images/50step2500neurontraining.png')
