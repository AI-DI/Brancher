import numpy as np
import matplotlib.pyplot as plt #TODO: WORK in progress!!

from brancher.variables import ProbabilisticModel
from brancher.stochastic_processes import MarkovProcess
from brancher.standard_variables import NormalVariable as Normal
from brancher.standard_variables import BetaVariable as Beta
from brancher.standard_variables import LogNormalVariable as LogNormal
from brancher.inference import perform_inference
from brancher.inference import ReverseKL

## Create latent time series ##
x0 = Normal(0, 0.5, "x_0")
X = MarkovProcess(x0, lambda t, x: Normal(x, 0.2, "x_{}".format(t)))

## Create observation model ##
Y = Normal(X, 1., "y")

## Sample ##
num_timepoints = 30
temporal_sample = Y.get_timeseries_sample(1, query_points=num_timepoints)
temporal_sample.plot()
plt.show()

## Observe model ##
data = temporal_sample
query_points = range(num_timepoints)
Y.observe(data, query_points)

## Variational model
Qx0 = Normal(0, 0.5, "x_0")
QX = [Qx0]
for idx in range(1, 30):
    QX.append(Normal(QX[idx-1], 0.25, "x_{}".format(idx), has_bias=True, learnable=True))
QX = ProbabilisticModel(QX)

## Perform ML inference ##
perform_inference(Y,
                  posterior_model=QX,
                  inference_method=ReverseKL(),
                  number_samples=50,
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