import numpy as np
import matplotlib.pyplot as plt

from brancher.variables import ProbabilisticModel
from brancher.stochastic_processes import MarkovProcess
from brancher.standard_variables import NormalVariable as Normal
from brancher.standard_variables import BetaVariable as Beta
from brancher.standard_variables import LogNormalVariable as LogNormal
from brancher.inference import perform_inference
from brancher.inference import ReverseKL, MAP

## Create time series model ##
x0 = Normal(0, 2, "x_0")
X = MarkovProcess(x0, lambda t, x: Normal(x, 0.25, "x_{}".format(t)))

## Sample ##
num_timepoints = 40
temporal_sample = X.get_timeseries_sample(1, query_points=num_timepoints)
temporal_sample.plot()
plt.show()

## Observe model ##
#data = temporal_sample
data = temporal_sample[:10].append(temporal_sample[30:])
#query_points = range(num_timepoints)
query_points = list(range(0, 10)) + list(range(30, 40))
X.observe(data, query_points)

## Variational model ##

# Variational parameters #

# Variational process #
#Qx0 = Normal(0, 1, "x_0", learnable=True)
#QX = MarkovProcess(Qx0, lambda t, x: Normal(0., 0.5, name="x_{}".format(t), has_bias=False, learnable=True))
Qx10 = Normal(float(temporal_sample[9:10].values), 0.25, "x_10")
QX = [Qx10]
for idx in range(11, 30):
    QX.append(Normal(QX[idx-11], 0.25, "x_{}".format(idx), has_bias=True, learnable=True))
QX = ProbabilisticModel(QX)

#X.set_posterior_model(process=QX)

## Perform ML inference ##
perform_inference(X,
                  posterior_model=QX,
                  inference_method=ReverseKL(),
                  number_iterations=3000,
                  number_samples=50,
                  optimizer="SGD",
                  lr=0.005)
loss_list = X.active_submodel.diagnostics["loss curve"]
plt.plot(loss_list)
plt.show()


## Sample ##
post_temporal_sample = X.get_posterior_timeseries_sample(100, query_points=100)
post_temporal_sample.plot(alpha=0.25)
temporal_sample.plot()
plt.show()
