
import matplotlib.pyplot as plt
import numpy as np

from brancher.variables import RootVariable, RandomVariable, ProbabilisticModel
from brancher.standard_variables import NormalVariable, LogNormalVariable, BetaVariable, MultivariateNormalVariable
from brancher import inference
import brancher.functions as BF

# Probabilistic model #
T = 50
driving_noise = 1.
measure_noise = 0.3
x0 = NormalVariable(0., driving_noise, 'x0')
y0 = NormalVariable(x0, measure_noise, 'y0')
b = BetaVariable(5., 5., 'b')

x = [x0]
y = [y0]
x_names = ["x0"]
y_names = ["y0"]
y_range = range(T)
for t in range(1, T):
    x_names.append("x{}".format(t))
    x.append(NormalVariable(b * x[t - 1], driving_noise, x_names[t]))
    if t in y_range:
        y_name = "y{}".format(t)
        y_names.append(y_name)
        y.append(NormalVariable(x[t], measure_noise, y_name))
AR_model = ProbabilisticModel(x + y)

# Generate data #
data = AR_model._get_sample(number_samples=1)
time_series = [float(data[yt].data) for yt in y]
ground_truth = [float(data[xt].data) for xt in x]
true_b = data[b].data
print("The true coefficient is: {}".format(float(true_b)))

# Observe data #
[yt.observe(data[yt][:, 0, :]) for yt in y]

# Structured variational distribution #
Qb = BetaVariable(5., 5., "b", learnable=True)
Qx = [NormalVariable(0., 1., 'x0', learnable=True)]
Qx_mean = [RootVariable(0., 'x0_mean', learnable=True)]
Qlambda = [RootVariable(0., 'x0_lambda', learnable=True)]


for t in range(1, T):
    Qx_mean.append(RootVariable(0., x_names[t] + "_mean", learnable=True))
    Qlambda.append(RootVariable(0., x_names[t] + "_lambda", learnable=True))
    Qx.append(NormalVariable(BF.sigmoid(Qlambda[t])*Qb*Qx[t - 1] + (1 - BF.sigmoid(Qlambda[t]))*Qx_mean[t], 1., x_names[t], learnable=True))
variational_posterior = ProbabilisticModel([Qb] + Qx)
AR_model.set_posterior_model(variational_posterior)

# Inference #
N_iter = 400
n_samples = 10
optimizer = "Adam"
lr = 0.01
inference.perform_inference(AR_model,
                            number_iterations=N_iter,
                            number_samples=n_samples,
                            optimizer=optimizer,
                            lr=lr)

loss_list = AR_model.diagnostics["loss curve"]

# ELBO
ELBO = AR_model.estimate_log_model_evidence(15000)
print("The ELBO is {}".format(ELBO))

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
ax1.scatter(y_range, time_series, c="k")
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

# Mean-field variational distribution #
Qb = BetaVariable(5., 5., "b", learnable=True)
Qx = [NormalVariable(0., 1., 'x0', learnable=True)]

for t in range(1, T):
    Qx.append(NormalVariable(0, 1., x_names[t], learnable=True))
variational_posterior = ProbabilisticModel([Qb] + Qx)
AR_model.set_posterior_model(variational_posterior)

# Inference #
inference.perform_inference(AR_model,
                            number_iterations=N_iter,
                            number_samples=n_samples,
                            optimizer=optimizer,
                            lr=lr)

loss_list = AR_model.diagnostics["loss curve"]

# ELBO
ELBO = AR_model.estimate_log_model_evidence(15000)
print("The ELBO is {}".format(ELBO))

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
ax1.scatter(y_range, time_series, c="k")
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


# # Multivariate normal variational distribution #
# QV = MultivariateNormalVariable(loc=np.zeros((T,)),
#                                 covariance_matrix=np.identity(T),
#                                 learnable=True)
# Qb = BetaVariable(8., 1., "b", learnable=True)
# Qx = [NormalVariable(QV[0], 0.1, 'x0', learnable=True)]
#
# for t in range(1, T):
#     Qx.append(NormalVariable(QV[t], 0.1, x_names[t], learnable=True))
# variational_posterior = ProbabilisticModel([Qb] + Qx)
# AR_model.set_posterior_model(variational_posterior)
#
# # Inference #
# inference.perform_inference(AR_model,
#                             number_iterations=N_iter,
#                             number_samples=n_samples,
#                             optimizer=optimizer,
#                             lr=lr)
#
# loss_list = AR_model.diagnostics["loss curve"]
#
# # ELBO
# ELBO = AR_model.estimate_log_model_evidence(15000)
# print("The ELBO is {}".format(ELBO))
#
# # Statistics
# posterior_samples = AR_model._get_posterior_sample(2000)
# b_posterior_samples = posterior_samples[b].detach().numpy().flatten()
# b_mean = np.mean(b_posterior_samples)
# b_sd = np.sqrt(np.var(b_posterior_samples))
#
# x_mean = []
# lower_bound = []
# upper_bound = []
# for xt in x:
#     x_posterior_samples = posterior_samples[xt].detach().numpy().flatten()
#     mean = np.mean(x_posterior_samples)
#     sd = np.sqrt(np.var(x_posterior_samples))
#     x_mean.append(mean)
#     lower_bound.append(mean - sd)
#     upper_bound.append(mean + sd)
# print("The estimated coefficient is: {} +- {}".format(b_mean, b_sd))
#
#
# # Two subplots, unpack the axes array immediately
# f, (ax1, ax2, ax3) = plt.subplots(1, 3)
# ax1.scatter(y_range, time_series, c="k")
# ax1.plot(range(T), x_mean)
# ax1.plot(range(T), ground_truth, c="k", ls ="--", lw=1.5)
# ax1.fill_between(range(T), lower_bound, upper_bound, alpha=0.5)
# ax1.set_title("Time series")
# ax2.plot(np.array(loss_list))
# ax2.set_title("Convergence")
# ax2.set_xlabel("Iteration")
# ax3.hist(b_posterior_samples, 25)
# ax3.axvline(x=true_b, lw=2, c="r")
# ax3.set_title("Posterior samples (b)")
# ax3.set_xlim(0, 1)
# plt.show()


