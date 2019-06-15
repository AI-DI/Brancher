import brancher.config as cfg
cfg.set_device("cpu")

import matplotlib.pyplot as plt
import numpy as np

from brancher.variables import ProbabilisticModel
from brancher.standard_variables import NormalVariable, LaplaceVariable, CauchyVariable, LogNormalVariable
from brancher import inference




# Real model
nu_real = 1.
mu_real = -2.
x_real = LaplaceVariable(mu_real, nu_real, "x_real")

# Normal model
nu = LogNormalVariable(0., 1., "nu")
mu = NormalVariable(0., 10., "mu")
x = LaplaceVariable(mu, nu, "x")
model = ProbabilisticModel([x])

# # Generate data
data = x_real._get_sample(number_samples=100)

# Observe data
x.observe(data[x_real][:, 0, :])

# Variational model
Qnu = LogNormalVariable(0., 1., "nu", learnable=True)
Qmu = NormalVariable(0., 1., "mu", learnable=True)
model.set_posterior_model(ProbabilisticModel([Qmu, Qnu]))

# Inference
inference.perform_inference(model,
                            number_iterations=3000,
                            number_samples=100,
                            optimizer='SGD',
                            lr=0.001)
loss_list = model.diagnostics["loss curve"]

# Statistics
posterior_samples = model._get_posterior_sample(5000)
nu_posterior_samples = posterior_samples[nu].cpu().detach().numpy().flatten()
mu_posterior_samples = posterior_samples[mu].cpu().detach().numpy().flatten()

# Two subplots, unpack the axes array immediately
f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
ax1.plot(np.array(loss_list))
ax1.set_title("Convergence")
ax1.set_xlabel("Iteration")
ax2.scatter(mu_posterior_samples, nu_posterior_samples, alpha=0.01)
ax2.scatter(mu_real, nu_real, c="r")
ax2.set_title("Posterior samples (b)")
ax3.hist(mu_posterior_samples, 25)
ax3.axvline(x=mu_real, lw=2, c="r")
ax4.hist(nu_posterior_samples, 25)
ax4.axvline(x=nu_real, lw=2, c="r")
plt.show()