# grid_cell_rnn
Based on the Sorscher et al. grid cell RNN, this model is meant to hold grids in its hidden layer across longer trajectories (~10,000 time steps). Lesions can be applied to assess the impact on grid cell translational and rotational drift.

Quick-start:
* lesion_analysis_pipeline.ipynb: apply lesions to a set of five models and track the translational and rotational drift in the grids over time
* 

Includes:

trajectory_generator.py: Generate simulated rat trajectories in a rectangular environment.

place_cells.py: Tile a set of simulated place cells across the training environment.

model.py: Contains the vanilla RNN model architecture, as well as an LSTM.

trainer.py: Contains model training loop.
