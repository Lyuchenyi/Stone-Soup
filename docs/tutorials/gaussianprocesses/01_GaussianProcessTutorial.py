#!/usr/bin/env python

"""
=========================================
1 - An introduction to Gaussian Processes
=========================================
"""

# %%
# This notebook is designed to introduce Gaussian Processes in Stone Soup using a single
# target scenario.
#
# Background and notation
# -----------------------
#
#

# %%
# A nearly-constant velocity example
# ----------------------------------
#
# The example is going to use a target following a constant velocity transition model.

import numpy as np
from datetime import datetime, timedelta

# %%
# Simulate a target
# ^^^^^^^^^^^^^^^^^
#
# We consider a 2d Cartesian scheme where the state vector is
# :math:`[x \ \dot{x} \ y \ \dot{y}]^T`.  That is, we'll model the target motion as a position
# and velocity component in each dimension. The units used are unimportant, but do need to be
# consistent.
#
# To start we'll create a simple truth path, sampling at 1
# second intervals. We'll do this by employing one of Stone Soup's native transition models.
#
# These inputs are required:
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
                                               ConstantVelocity

# And the clock starts
start_time = datetime.now().replace(second=0, microsecond=0)
np.random.seed(1)

# %%
# We want a 2d simulation, so we create a constant velocity transition model:
q_x = 0.05
q_y = 0.05
transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(q_x),
                                                          ConstantVelocity(q_y)])

# %%
# We can now simulate a ground truth starting (0,0) and moving to the north east.
timesteps = [start_time]
truth = GroundTruthPath([GroundTruthState([0, 1, 0, 1], timestamp=timesteps[0])])

num_steps = 20
for k in range(1, num_steps + 1):

    timesteps.append(start_time+timedelta(seconds=k))  # add next timestep to list of timesteps
    truth.append(GroundTruthState(
        transition_model.function(truth[k-1], noise=True, time_interval=timedelta(seconds=1)),
        timestamp=timesteps[k]))

# %%
# We now have ground truth is generated and we can plot the result.

from stonesoup.plotter import Plotter
plotter = Plotter()
plotter.plot_ground_truths(truth, [0, 2])
plotter.fig

# %%
# Simulate measurements
# ^^^^^^^^^^^^^^^^^^^^^
#
# We will now simulate measurements from a sensor that follow a linear measurement model that
# detects the position of a target, such that
# :math:`\mathbf{z}_k = H_k \mathbf{x}_k + \boldsymbol{\nu}_k`,
# :math:`\boldsymbol{\nu}_k \sim \mathcal{N}(0,R)`, with
#
# .. math::
#           H_k &= \begin{bmatrix}
#                     1 & 0 & 0 & 0\\
#                     0  & 0 & 1 & 0\\
#                       \end{bmatrix}\\
#           R &= \begin{bmatrix}
#                   1 & 0\\
#                     0 & 1\\
#                \end{bmatrix} \omega
#
# where :math:`\omega` is set to 5.

from stonesoup.types.detection import Detection
from stonesoup.models.measurement.linear import LinearGaussian

# %%
# The linear Gaussian measurement model is set up by indicating the number of dimensions in the
# state vector and the dimensions that are measured (so specifying :math:`H_k`) and the noise
# covariance matrix :math:`R`.
measurement_model = LinearGaussian(
    ndim_state=4,  # Number of state dimensions (position and velocity in 2D)
    mapping=(0, 2),  # Mapping measurement vector index to state index
    noise_covar=np.array([[5, 0],  # Covariance matrix for Gaussian PDF
                          [0, 5]])
    )

# %%
# Generate the measurements
measurements = []
for state in truth:
    measurement = measurement_model.function(state, noise=True)
    measurements.append(Detection(measurement,
                                  timestamp=state.timestamp,
                                  measurement_model=measurement_model))

# %%
# Plot the result, again mapping the x and y position values
plotter.plot_measurements(measurements, [0, 2])
plotter.fig

# %%
# Construct a Gaussian Process
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
