import chainer
import matplotlib.pyplot as plt
import numpy as np

from brancher.variables import RootVariable, RandomVariable, ProbabilisticModel
from brancher.standard_variables import NormalVariable, LogNormalVariable, BetaVariable
from brancher import inference
import brancher.functions as BF

# Probabilistic model #
T = 30
driving_noise = 1.
measure_noise = 0.5
x0 = NormalVariable(0., driving_noise, 'x0')
y0 = NormalVariable(x0, measure_noise, 'x0')
b = BetaVariable(1., 1., 'b')

x = [x0]
y = [y0]
x_names = ["x0"]
y_names = ["y0"]
for t in range(1,T):
    x_names.append("x{}".format(t))
    y_names.append("y{}".format(t))
    x.append(NormalVariable(b*x[t-1], driving_noise, x_names[t]))
    y.append(NormalVariable(x[t], measure_noise, y_names[t]))
AR_model = ProbabilisticModel(x + y)

# Generate data #
data = AR_model._get_sample(number_samples=1)
time_series = [float(data[yt].data) for yt in y]
ground_truth = [float(data[xt].data) for xt in x]
true_b = data[b].data
print("The true coefficient is: {}".format(float(true_b)))

# Observe data #
[yt.observe(data[yt][:, 0, :]) for yt in y]

# Autoregressive variational distribution #
Qb = BetaVariable(1., 1., "b", learnable=True)
logit_b_post = RootVariable(0., 'logit_b_post', learnable=True)
Qx = [NormalVariable(0., 1., 'x0', learnable=True)]
Qx_mean = [RootVariable(0., 'x0_mean', learnable=True)]
for t in range(1, T):
    Qx_mean.append(RootVariable(0., x_names[t] + "_mean", learnable=True))
    Qx.append(NormalVariable(logit_b_post*Qx[t-1] + Qx_mean[t], 1., x_names[t], learnable=True))
variational_posterior = ProbabilisticModel([Qb] + Qx)
AR_model.set_posterior_model(variational_posterior)

# Inference #
inference.perform_inference(AR_model,
                            number_iterations=200,
                            number_samples=100,
                            optimizer='Adam',
                            lr=0.01)

loss_list = AR_model.diagnostics["loss curve"]

# Statistics
posterior_samples = AR_model._get_posterior_sample(2000)
b_posterior_samples = posterior_samples[b].detach().numpy().flatten()
b_mean = np.mean(b_posterior_samples)
b_sd = np.sqrt(np.var(b_posterior_samples))

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
print("The estimated coefficient is: {} +- {}".format(b_mean, b_sd))


# Two subplots, unpack the axes array immediately
f, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.scatter(range(T), time_series, c="k")
ax1.plot(range(T), x_mean)
ax1.plot(range(T), ground_truth, c="k", ls ="--", lw=1.5)
ax1.fill_between(range(T), lower_bound, upper_bound, alpha=0.5)
ax1.set_title("Time series")
ax2.plot(np.array(loss_list))
ax2.set_title("Convergence")
ax2.set_xlabel("Iteration")
ax3.hist(b_posterior_samples, 25)
ax3.axvline(x=true_b, lw=2, c="r")
ax3.set_title("Posterior samples (b)")
ax3.set_xlim(0, 1)
plt.show()