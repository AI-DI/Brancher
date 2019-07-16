import numpy as np
import matplotlib.pyplot as plt

from brancher.stochastic_processes import MarkovProcess
from brancher.standard_variables import NormalStandardVariable as Normal
from brancher.standard_variables import BetaStandardVariable as Beta
from brancher.standard_variables import LogNormalStandardVariable as LogNormal
from brancher.inference import perform_inference
from brancher.inference import ReverseKL

## Create time series model ##
x0 = Normal(0, 1, "x_0")
X = MarkovProcess(x0, lambda t, x: Normal(x, 0.6, "x_{}".format(t)))

## Sample ##
num_timepoints = 100
temporal_sample = X.get_timeseries_sample(1, query_points=num_timepoints)
temporal_sample.plot()
plt.show()

## Observe model ##
#data = temporal_sample
data = temporal_sample[:10].append(temporal_sample[90:])
#query_points = range(num_timepoints)
query_points = list(range(0, 10)) + list(range(90, 100))
X.observe(data, query_points)

## Variational model ##

# Variational parameters #

# Variational process #
Qx0 = Normal(0, 1, "x_0", learnable=True)
Qbeta = Beta(1, 1, "beta", learnable=True)
QX = MarkovProcess(Qx0, lambda t, x: Normal(Qbeta*x, 0.6, name="x_{}".format(t), has_bias=True, learnable=True))

X.set_posterior_model(process=QX)

## Perform ML inference ##
perform_inference(X,
                  inference_method=ReverseKL(),
                  number_iterations=1000,
                  number_samples=100,
                  optimizer="Adagrad",
                  lr=0.01)
loss_list = X.active_submodel.diagnostics["loss curve"]
plt.plot(loss_list)
plt.show()


## Sample ##
post_temporal_sample = X.get_posterior_timeseries_sample(20, query_points=100)
post_temporal_sample.plot()
plt.show()
