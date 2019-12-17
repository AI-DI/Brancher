import matplotlib.pyplot as plt
import numpy as np
import torch

from brancher.variables import ProbabilisticModel
from brancher.standard_variables import NormalVariable, DeterministicVariable, LogNormalVariable
import brancher.functions as BF
from brancher.visualizations import plot_density
from brancher.transformations import PlanarFlow
from brancher import inference
from brancher.visualizations import plot_posterior



# Model
M = 8
y = NormalVariable(torch.zeros((M,)), 1.*torch.ones((M,)), "y")
y0 = DeterministicVariable(y[1], "y0")
d = NormalVariable(y, torch.ones((M,)), "d")
model = ProbabilisticModel([d, y, y0])

# get samples
d.observe(d.get_sample(55, input_values={y: 1.*torch.ones((M,))}))

# Variational distribution
u1 = DeterministicVariable(torch.normal(0., 1., (M, 1)), "u1", learnable=True)
w1 = DeterministicVariable(torch.normal(0., 1., (M, 1)), "w1", learnable=True)
b1 = DeterministicVariable(torch.normal(0., 1., (1, 1)), "b1", learnable=True)
u2 = DeterministicVariable(torch.normal(0., 1., (M, 1)), "u2", learnable=True)
w2 = DeterministicVariable(torch.normal(0., 1., (M, 1)), "w2", learnable=True)
b2 = DeterministicVariable(torch.normal(0., 1., (1, 1)), "b2", learnable=True)
z = NormalVariable(torch.zeros((M, 1)), torch.ones((M, 1)), "z", learnable=True)
Qy = PlanarFlow(w2, u2, b2)(PlanarFlow(w1, u1, b1)(z))
Qy.name = "y"
Qy0 = DeterministicVariable(Qy[1], "y0")

#Qy._get_sample(4)[Qy].shape

variational_model = ProbabilisticModel([Qy, Qy0])
model.set_posterior_model(variational_model)

# Inference #
inference.perform_inference(model,
                            number_iterations=400,
                            number_samples=100,
                            optimizer="Adam",
                            lr=0.5)

loss_list1 = model.diagnostics["loss curve"]

#Plot posterior
plot_posterior(model, variables=["y0"])
plt.show()

# Variational distribution
Qy = NormalVariable(torch.zeros((M,)), 0.5*torch.ones((M,)), "y", learnable=True)
Qy0 = DeterministicVariable(Qy[1], "y0")

variational_model = ProbabilisticModel([Qy, Qy0])
model.set_posterior_model(variational_model)

# Inference #
inference.perform_inference(model,
                            number_iterations=400,
                            number_samples=100,
                            optimizer="Adam",
                            lr=0.01)

loss_list2 = model.diagnostics["loss curve"]

#Plot posterior
plot_posterior(model, variables=["y0"])
plt.show()

plt.plot(loss_list1)
plt.plot(loss_list2)
plt.show()


