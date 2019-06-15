import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from brancher.variables import ProbabilisticModel

from brancher.stochastic_processes import GaussianProcess as GP
from brancher.stochastic_processes import SquaredExponentialCovariance as SquaredExponential
from brancher.stochastic_processes import ConstantMean
from brancher.variables import RootVariable
from brancher.standard_variables import NormalVariable as Normal
from brancher import inference

num_datapoints = 20
x_range = np.linspace(-2, 2, num_datapoints)
x = RootVariable(x_range, name="x")

# Model
mu = ConstantMean(0.)
cov = SquaredExponential(scale=0.2, jitter=10**-4)
f = GP(mu, cov, name="f")
y = Normal(f(x), 0.2, name="y")
model = ProbabilisticModel([y])

# Observe data
noise_level = 0.2
data = np.sin(2*np.pi*0.4*x_range) + noise_level*np.random.normal(0., 1., (1, num_datapoints))
y.observe(data)

#Variational Model
Qf = Normal(loc=np.zeros((num_datapoints,)),
            scale=2.,
            name="f(x)",
            learnable=True)
variational_model = ProbabilisticModel([Qf])
model.set_posterior_model(variational_model)

# Inference
inference.perform_inference(model,
                            number_iterations=2000,
                            number_samples=20,
                            optimizer='SGD',
                            lr=0.00001)
loss_list = model.diagnostics["loss curve"]
plt.plot(loss_list)
plt.show()

# Posterior
posterior_samples = model.get_posterior_sample(8000)["f(x)"]
posterior_mean = posterior_samples.mean()
plt.plot(x_range, posterior_mean)
plt.scatter(x_range, data, color="k")
plt.show()