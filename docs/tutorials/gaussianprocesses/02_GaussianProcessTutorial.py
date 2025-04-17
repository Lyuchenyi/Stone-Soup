#!/usr/bin/env python

"""
============================================================================
2 - An introduction to Gaussian Processes with Gaussian Process ground truth
============================================================================
"""

# %%
# This notebook is designed to follow on from the introduction to Gaussian Processes tutorial
# using a single target scenario where the target's motion is following a Gaussian Process.
#
# Background and notation
# -----------------------
#
#

# %%
# A Gaussian Process derived trajectory example
# ---------------------------------------------
#
# The example is going to use a target following a Gaussian Process motion.

import numpy as np
from datetime import datetime

# %%
# Simulate a target
# ^^^^^^^^^^^^^^^^^
#

from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

start_time = datetime.now().replace(second=0, microsecond=0)
np.random.seed(1)
# %%
# We can now simulate a ground truth starting (0,0) and moving to the north east.
truth = GroundTruthPath()

num_steps = 20
for k in range(1, num_steps + 1):
    truth.append(GroundTruthState())

# %%
# We now have ground truth is generated and we can plot the result.

from stonesoup.plotter import Plotter
plotter = Plotter()
plotter.plot_ground_truths(truth, [0, 1])
plotter.fig

# %%
# Simulate measurements
# ^^^^^^^^^^^^^^^^^^^^^
#
# We will now simulate measurements from a sensor that follow a linear measurement model that
# detects the position of a target, such that


from stonesoup.types.detection import Detection
from stonesoup.models.measurement.linear import LinearGaussian

# %%
# The linear Gaussian measurement model is set up by indicating the number of dimensions in the
# state vector and the dimensions that are measured (so specifying :math:`H_k`) and the noise
# covariance matrix :math:`R`.
measurement_model = LinearGaussian(
    ndim_state=2,  # Number of state dimensions (position and velocity in 2D)
    mapping=(0, 1),  # Mapping measurement vector index to state index
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
# We now use the same Gaussian Process method as in tutorial 1.
