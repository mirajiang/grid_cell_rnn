# grid_cell_rnn
Based on the Sorscher et al. grid cell RNN, this model is meant to hold grids in its hidden layer across longer trajectories (~10,000 time steps). Lesions can be applied to assess the impact on grid cell translational and rotational drift.

Quick-start:
* lesion_analysis_pipeline.ipynb: Apply lesions to a set of trained models and track the translational and rotational drift in the grids over time.
* training_models.py: Run this script to train a new RNN model.
* grid_cell_analysis: Probe the 

Includes:
* trajectory_generator.py: Generate simulated rat trajectories in a rectangular environment.
* place_cells.py: Tile a set of simulated place cells across the training environment.
* model.py: Contains the RNN model architecture with options to apply lesions
* trainer.py: Contains model training loop.
* utils.py: Contains support functions for plotting ratemaps, computing grid scores, and tracking translational/rotational drift in the grid cells over time
