import numpy as np
import matplotlib.pyplot as plt

from brancher.stochastic_processes import MarkovProcess
from brancher.standard_variables import NormalVariable as Normal
from brancher.standard_variables import BetaVariable as Beta
from brancher.standard_variables import LogNormalVariable as LogNormal
from brancher.inference import perform_inference
from brancher.inference import ReverseKL

## Create latent time series ##
x0 = Normal(0, 1, "x_0")
b = Beta(1, 1, "b")
sigma = LogNormal(0, 1, "sigma")
X = MarkovProcess(x0, lambda x: Normal(b*x, sigma, "x"))

## Create observation model ##
chi = LogNormal(1, 0.5, "sigma")
Y = Normal(X, chi, "y")

## Sample ##
num_timepoints = 20
temporal_sample = Y.get_timeseries_sample(1, query_points=num_timepoints,
                                          input_values={sigma: 1., b: 1.})
temporal_sample.plot()
plt.show()

## Observe model ##
data = temporal_sample
query_points = range(num_timepoints)
Y.observe(data, query_points)

## Perform ML inference ##
perform_inference(Y,
                  inference_method=ReverseKL(),
                  number_iterations=1000,
                  optimizer="SGD",
                  lr=0.005)
loss_list = Y.active_submodel.diagnostics["loss curve"]
plt.plot(loss_list)
plt.show()


## Sample ##
post_temporal_sample = Y.get_posterior_timeseries_sample(20, query_points=100)
post_temporal_sample.plot()
plt.show()
print("Done")