import matplotlib.pyplot as plt
import numpy as np
import torch

from brancher.variables import ProbabilisticModel
from brancher.standard_variables import NormalVariable, DeterministicVariable, BetaVariable
import brancher.functions as BF
from brancher.visualizations import plot_density
from brancher.transformations import Exp, Scaling, TriangularLinear, Sigmoid, Bias, PlanarFlow
from brancher import inference
from brancher.visualizations import plot_posterior



# Probabilistic model #
T = 30
driving_noise = 1.
measure_noise = 0.5
x0 = NormalVariable(0., driving_noise, 'x0')
y0 = NormalVariable(x0, measure_noise, 'y0')
b = 0.8

x = [x0]
y = [y0]
x_names = ["x0"]
y_names = ["y0"]
for t in range(1,T):
    x_names.append("x{}".format(t))
    y_names.append("y{}".format(t))
    x.append(NormalVariable(b * x[t - 1], driving_noise, x_names[t]))
    y.append(NormalVariable(x[t], measure_noise, y_names[t]))
AR_model = ProbabilisticModel(x + y)

# Generate data #
data = AR_model._get_sample(number_samples=1)
time_series = [float(data[yt].data) for yt in y]
ground_truth = [float(data[xt].data) for xt in x]

# Observe data #
[yt.observe(data[yt][:, 0, :]) for yt in y]

# Variational distribution
# N = int(T*(T+1)/2)
# v1 = DeterministicVariable(torch.normal(0., 0.1, (N,)), "v1", learnable=True)
# v2 = DeterministicVariable(torch.normal(0., 0.1, (N,)), "v2", learnable=True)
# b1 = DeterministicVariable(torch.normal(0., 0.1, (T,1)), "b1", learnable=True)
# w1 = DeterministicVariable(torch.normal(0., 0.1, (N,)), "w1", learnable=True)
# w2 = DeterministicVariable(torch.normal(0., 0.1, (N,)), "w2", learnable=True)
# b2 = DeterministicVariable(torch.normal(0., 0.1, (T,1)), "b2", learnable=True)
# Qz = NormalVariable(torch.zeros((T, 1)), torch.ones((T, 1)), "z")
# Qtrz = Bias(b1)(TriangularLinear(w1, T)(TriangularLinear(w2, T, upper=True)(Sigmoid()(Bias(b2)(TriangularLinear(v1, T)(TriangularLinear(v2, T, upper=True)(Qz)))))))

# Variational distribution
u1 = DeterministicVariable(torch.normal(0., 1., (T, 1)), "u1", learnable=True)
w1 = DeterministicVariable(torch.normal(0., 1., (T, 1)), "w1", learnable=True)
b1 = DeterministicVariable(torch.normal(0., 1., (1, 1)), "b1", learnable=True)
u2 = DeterministicVariable(torch.normal(0., 1., (T, 1)), "u2", learnable=True)
w2 = DeterministicVariable(torch.normal(0., 1., (T, 1)), "w2", learnable=True)
b2 = DeterministicVariable(torch.normal(0., 1., (1, 1)), "b2", learnable=True)
z = NormalVariable(torch.zeros((T, 1)), torch.ones((T, 1)), "z", learnable=True)
Qtrz = PlanarFlow(w2, u2, b2)(PlanarFlow(w1, u1, b1)(z))

Qx = []
for t in range(0, T):
    Qx.append(DeterministicVariable(Qtrz[t], name=x_names[t]))

variational_model = ProbabilisticModel(Qx)
AR_model.set_posterior_model(variational_model)

# Inference #
inference.perform_inference(AR_model,
                            number_iterations=300,
                            number_samples=50,
                            optimizer="Adam",
                            lr=0.01)

loss_list = AR_model.diagnostics["loss curve"]

# Statistics
posterior_samples = AR_model._get_posterior_sample(2000)

x_mean = []
lower_bound = []
upper_bound = []
for xt in x:
    x_posterior_samples = posterior_samples[xt].detach().numpy().flatten()
    mean = np.mean(x_posterior_samples)
    sd = np.sqrt(np.var(x_posterior_samples))
    x_mean.append(mean)
    lower_bound.append(mean - sd)
    upper_bound.append(mean + sd)


# Two subplots, unpack the axes array immediately
f, (ax1, ax2) = plt.subplots(1, 2)
ax1.scatter(range(T), time_series, c="k")
ax1.plot(range(T), x_mean)
ax1.plot(range(T), ground_truth, c="k", ls ="--", lw=1.5)
ax1.fill_between(range(T), lower_bound, upper_bound, alpha=0.5)
ax1.set_title("Time series")
ax2.plot(np.array(loss_list))
ax2.set_title("Convergence")
ax2.set_xlabel("Iteration")
plt.show()


