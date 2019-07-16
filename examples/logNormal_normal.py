import brancher.config as cfg
cfg.set_device("cpu")

import matplotlib.pyplot as plt
import numpy as np

from brancher.variables import ProbabilisticModel
from brancher.standard_variables import NormalStandardVariable, LogNormalStandardVariable
from brancher import inference

# Normal model
nu = LogNormalStandardVariable(0., 1., "nu")
mu = NormalStandardVariable(0., 10., "mu")
x = NormalStandardVariable(mu, nu, "x")
model = ProbabilisticModel([x]) # to fix plot_posterior (flatten automatically?)

# # Generate data
nu_real = 1.
mu_real = -2.
data = model.get_sample(number_samples=20, input_values={mu: mu_real, nu: nu_real})

# Observe data
x.observe(data)

# Variational model
Qnu = LogNormalStandardVariable(0., 1., "nu", learnable=True)
Qmu = NormalStandardVariable(0., 1., "mu", learnable=True)
model.set_posterior_model(ProbabilisticModel([Qmu, Qnu]))

# Inference
inference.perform_inference(model,
                            number_iterations=300,
                            number_samples=100,
                            optimizer='SGD',
                            lr=0.0001)
loss_list = model.diagnostics["loss curve"]

plt.plot(loss_list)
plt.title("Loss (negative ELBO)")
plt.show()

from brancher.visualizations import plot_posterior

plot_posterior(model, variables=["mu", "nu", "x"])
plt.show()