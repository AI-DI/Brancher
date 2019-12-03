import matplotlib.pyplot as plt
import numpy as np
import torch

from brancher.variables import ProbabilisticModel
from brancher.standard_variables import NormalVariable, DeterministicVariable, LogNormalVariable
import brancher.functions as BF
from brancher.visualizations import plot_density
from brancher.transformations import Exp, Scaling, TriangularLinear, Sigmoid, Bias
from brancher import inference
from brancher.visualizations import plot_posterior



# Model
M = 3
y = NormalVariable(torch.zeros((M,)), 1.*torch.ones((M,)), "y")
y0 = DeterministicVariable(y[0], "y0")
d = NormalVariable(y**2, torch.ones((M,)), "d")
model = ProbabilisticModel([d, y, y0])

# get samples
d.observe(d.get_sample(25, input_values={y: 0.3*torch.ones((M,))}))

# Variational distribution
N = int(M*(M+1)/2)
v1 = DeterministicVariable(torch.normal(0., 0.1, (N,)), "v1", learnable=True)
v2 = DeterministicVariable(torch.normal(0., 0.1, (N,)), "v2", learnable=True)
b1 = DeterministicVariable(torch.normal(0., 0.1, (M,1)), "b1", learnable=True)
w1 = DeterministicVariable(torch.normal(0., 0.1, (N,)), "w1", learnable=True)
w2 = DeterministicVariable(torch.normal(0., 0.1, (N,)), "w2", learnable=True)
b2 = DeterministicVariable(torch.normal(0., 0.1, (M,1)), "b2", learnable=True)
z = NormalVariable(torch.zeros((M, 1)), torch.ones((M, 1)), "z")
Qy = Bias(b1)(TriangularLinear(w1, M)(TriangularLinear(w2, M, upper=True)(Sigmoid()(Bias(b2)(TriangularLinear(v1, M)(TriangularLinear(v2, M, upper=True)(z)))))))
Qy.name = "y"
Qy0 = DeterministicVariable(Qy[0], "y0")

variational_model = ProbabilisticModel([Qy, Qy0])
model.set_posterior_model(variational_model)

# Inference #
inference.perform_inference(model,
                            number_iterations=1500,
                            number_samples=50,
                            optimizer="Adam",
                            lr=0.001)

loss_list1 = model.diagnostics["loss curve"]

#Plot posterior
plot_posterior(model, variables=["y0"])
plt.show()

# Variational distribution
Qy = NormalVariable(torch.zeros((M,)), 0.5*torch.ones((M,)), "y", learnable=True)
Qy0 = DeterministicVariable(Qy[0], "y0")

variational_model = ProbabilisticModel([Qy, Qy0])
model.set_posterior_model(variational_model)

# Inference #
inference.perform_inference(model,
                            number_iterations=1500,
                            number_samples=50,
                            optimizer="Adam",
                            lr=0.01)

loss_list2 = model.diagnostics["loss curve"]

#Plot posterior
plot_posterior(model, variables=["y0"])
plt.show()

plt.plot(loss_list1)
plt.plot(loss_list2)
plt.show()


