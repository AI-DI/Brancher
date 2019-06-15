import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from brancher.variables import ProbabilisticModel

from brancher.stochastic_processes import GaussianProcess as GP
from brancher.stochastic_processes import SquaredExponentialCovariance as SquaredExponential
from brancher.stochastic_processes import WhiteNoiseCovariance as WhiteNoise
from brancher.stochastic_processes import HarmonicCovariance as Harmonic
from brancher.stochastic_processes import ConstantMean
from brancher.variables import RootVariable
from brancher.standard_variables import NormalVariable as Normal
from brancher.standard_variables import LogNormalVariable as LogNormal
from brancher.standard_variables import MultivariateNormalVariable as MultivariateNormal
from brancher import inference
from brancher.visualizations import plot_posterior
import brancher.functions as BF

num_datapoints = 60
x_range = np.linspace(-1, 1, num_datapoints)
x = RootVariable(x_range, name="x")

# Model
length_scale = LogNormal(0., 0.3, name="length_scale")
noise_var = LogNormal(0., 0.3, name="noise_var")
freq = Normal(0.5, 0.5, name="freq")
mu = ConstantMean(0.5)
cov = Harmonic(frequency=freq)*SquaredExponential(scale=length_scale) + WhiteNoise(magnitude=noise_var)
f = GP(mu, cov, name="f")
y = f(x)
model = ProbabilisticModel([y])

# Observe data
noise_level = 0.2
f1 = 1.
data = np.sin(2*np.pi*f1*x_range) + noise_level*np.random.normal(0., 1., (1, num_datapoints))
y.observe(data)

#Variational Model
Qlength_scale = LogNormal(-1, 0.2, name="length_scale", learnable=True)
Qnoise_var = LogNormal(-1, 0.2, name="noise_var", learnable=True)
Qfreq = Normal(0.2, 0.2, name="freq", learnable=True)
variational_model = ProbabilisticModel([Qlength_scale, Qnoise_var, Qfreq])
model.set_posterior_model(variational_model)

# Inference
inference.perform_inference(model,
                            number_iterations=1500,
                            number_samples=10,
                            optimizer='SGD',
                            lr=0.0025)
loss_list = model.diagnostics["loss curve"]
plt.plot(loss_list)
plt.show()

# Posterior plot
plot_posterior(model, variables=["length_scale", "noise_var", "freq"])
plt.show()