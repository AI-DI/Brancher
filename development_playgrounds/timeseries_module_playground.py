import numpy as np
import matplotlib.pyplot as plt

from brancher.stochastic_processes import MarkovProcess
from brancher.standard_variables import NormalVariable as Normal
from brancher.standard_variables import BetaVariable as Beta
from brancher.standard_variables import LogNormalVariable as LogNormal
from brancher.inference import perform_inference
from brancher.inference import MAP

## Create time series model ##
x0 = Normal(0, 1, "x_0")
b = Beta(1, 1, "b")
sigma = LogNormal(1, 1, "sigma")
X = MarkovProcess(x0, lambda x: Normal(b*x, sigma, "x"))

## Sample ##
num_timepoints = 50
temporal_sample = X.get_timeseries_sample(1, query_points=num_timepoints,
                                          input_values={sigma: 1., b: 1.})
temporal_sample.plot()
plt.show()

## Observe model ##
data = temporal_sample
query_points = range(num_timepoints)
X.observe(data, query_points)


## Perform ML inference ##
perform_inference(X,
                  inference_method=MAP(),
                  number_iterations=1000,
                  optimizer="Adam",
                  lr=0.01)
loss_list = X.active_submodel.diagnostics["loss curve"]
plt.plot(loss_list)
plt.show()


## Sample ##
post_temporal_sample = X.get_posterior_timeseries_sample(20, query_points=100)
post_temporal_sample.plot()
plt.show()
