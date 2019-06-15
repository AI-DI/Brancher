import chainer

from brancher.variables import ProbabilisticModel
from brancher.standard_variables import NormalVariable, LogNormalVariable
from brancher import inference

# Real model
nu_real = 1.
mu_real = -2.
x_real = NormalVariable(mu_real, nu_real, "x_real")

# Normal model
nu = LogNormalVariable(0., 1., "nu")
mu = NormalVariable(0., 10., "mu")
x = NormalVariable(mu, nu, "x")
model = ProbabilisticModel([x])

print(model)

# Print samples
sample = model.get_sample(10)
print(sample)

# Print samples from single variable
x_sample = x.get_sample(10)
print(x_sample)

# Print samples conditional on an input
in_sample = model.get_sample(10, input_values={mu: 100.})
print(in_sample)

# # Generate data
data = x_real._get_sample(number_samples=50)

# Observe data
x.observe(data[x_real][:, 0, :])

# Variational model
Qnu = LogNormalVariable(0., 1., "nu", learnable=True)
Qmu = NormalVariable(0., 1., "mu", learnable=True)
model.set_posterior_model(ProbabilisticModel([Qmu, Qnu]))

# Inference
inference.perform_inference(model,
                            number_iterations=100,
                            number_samples=50,
                            optimizer=chainer.optimizers.Adam(0.1))
loss_list = model.diagnostics["loss curve"]

# print posterior sample
post_samples = model.get_posterior_sample(10)
print(post_samples)