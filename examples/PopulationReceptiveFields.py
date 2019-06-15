import matplotlib.pyplot as plt
import numpy as np

from brancher.variables import RootVariable, RandomVariable, ProbabilisticModel
from brancher.standard_variables import NormalVariable, LogNormalVariable
from brancher import functions as BF
from brancher import inference
from brancher.visualizations import plot_posterior

# Parameters
S = 6.
N = 40
x_range = np.linspace(-S/2., S/2., N)
y_range = np.linspace(-S/2., S/2., N)
x_mesh, y_mesh = np.meshgrid(x_range, y_range)
#x_mesh, y_mesh = np.expand_dims(x_mesh, 0), np.expand_dims(y_mesh, 0)

# Experimental model
x = RootVariable(x_mesh, name="x") #TODO: it should create this automatically
y = RootVariable(y_mesh, name="y")
w1 = NormalVariable(0., 1., name="w1")
w2 = NormalVariable(0., 1., name="w2")
b = NormalVariable(0., 1., name="b")
experimental_input = NormalVariable(BF.exp(BF.sin(w1*x + w2*y + b)), 0.1, name="input", is_observed=True)

# Probabilistic Model
mu_x = NormalVariable(0., 1., name="mu_x")
mu_y = NormalVariable(0., 1., name="mu_y")
v = LogNormalVariable(0., 0.1, name="v")
nu = LogNormalVariable(-1, 0.01, name="nu")
receptive_field = BF.exp((-(x - mu_x)**2 - (y - mu_y)**2)/(2.*v**2))/(2.*BF.sqrt(np.pi*v**2))
mean_response = BF.sum(BF.sum(receptive_field*experimental_input, dim=1, keepdim=True), dim=2, keepdim=True) #TODO; not very intuitive
response = NormalVariable(mean_response, nu, name="response")
model = ProbabilisticModel([response, experimental_input])

# Generate data and observe the model
sample = model.get_sample(15, input_values={mu_x: 1., mu_y: 2., v: 0.3, nu: 0.1})[["x", "y", "w1", "w2", "b", "response"]]
model.observe(sample)

# Variational model
Qmu_x = NormalVariable(0., 1., name="mu_x", learnable=True)
Qmu_y = NormalVariable(0., 1., name="mu_y", learnable=True)
Qv = LogNormalVariable(0., 0.1, name="v", learnable=True)
Qnu = LogNormalVariable(-1, 0.01, name="nu", learnable=True)
variational_posterior = ProbabilisticModel([Qmu_x, Qmu_y, Qv, Qnu])
model.set_posterior_model(variational_posterior)

# Inference #
inference.perform_inference(model,
                            number_iterations=1500,
                            number_samples=50,
                            optimizer='Adam',
                            lr=0.01)
loss_list = model.diagnostics["loss curve"]
plt.plot(loss_list)
plt.show()

# Plot posterior
plot_posterior(model, variables=["mu_x", "mu_y", "v"])
plt.show()

