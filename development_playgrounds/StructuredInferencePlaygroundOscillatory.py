import chainer
import matplotlib.pyplot as plt
import numpy as np

from brancher.variables import RootVariable, RandomVariable, ProbabilisticModel
from brancher.standard_variables import NormalVariable, LogNormalVariable, BetaVariable, MultivariateNormalVariable
from brancher import inference
import brancher.functions as BF

# Probabilistic model #
T = 50
dt = 0.01
driving_noise = 0.5
measure_noise = 0.2
x0 = NormalVariable(0., driving_noise, 'x0')
y0 = NormalVariable(x0, measure_noise, 'y0')
x1 = NormalVariable(0., driving_noise, 'x1')
y1 = NormalVariable(x1, measure_noise, 'y1')
b = 25
omega = NormalVariable(2*np.pi*8, 1., 'omega')

x = [x0, x1]
y = [y0, y1]
x_names = ["x0", "x1"]
y_names = ["y0", "y1"]
y_range = [t for t in range(T) if (t < 20 or t > 40)]
for t in range(2, T):
    x_names.append("x{}".format(t))
    new_mu = (-1 - omega**2*dt**2 + b*dt)*x[t - 2] + (2 - b*dt)*x[t - 1]
    x.append(NormalVariable(new_mu, np.sqrt(dt)*driving_noise, x_names[t]))
    if t in y_range:
        y_name = "y{}".format(t)
        y_names.append(y_name)
        y.append(NormalVariable(x[t], measure_noise, y_name))
AR_model = ProbabilisticModel(x + y)

# Generate data #
data = AR_model._get_sample(number_samples=1)
time_series = [float(data[yt].data) for yt in y]
ground_truth = [float(data[xt].data) for xt in x]
true_b = data[omega].data
print("The true coefficient is: {}".format(float(true_b)))

# Observe data #
[yt.observe(data[yt][:, 0, :]) for yt in y]

# get time series
plt.plot([data[xt][:, 0, :] for xt in x])
plt.scatter(y_range, time_series, c="k")
plt.show()


# Structured variational distribution #
Qomega = NormalVariable(2*np.pi*8, 1., 'omega', learnable=True)
Qx = [NormalVariable(0., 1., 'x0', learnable=True),
      NormalVariable(0., 1., 'x1', learnable=True)]
Qx_mean = [RootVariable(0., 'x0_mean', learnable=True),
           RootVariable(0., 'x1_mean', learnable=True)]
Qlambda = [RootVariable(-0.5, 'x0_lambda', learnable=True),
           RootVariable(-0.5, 'x1_lambda', learnable=True)]


for t in range(2, T):
    if t in y_range:
        l = -0.5
    else:
        l = 0.5
    Qx_mean.append(RootVariable(0, x_names[t] + "_mean", learnable=True))
    Qlambda.append(RootVariable(l, x_names[t] + "_lambda", learnable=True))
    new_mu = (-1 - Qomega ** 2 * dt ** 2 + b * dt) * Qx[t - 2] + (2 - b * dt) * Qx[t - 1]
    Qx.append(NormalVariable(BF.sigmoid(Qlambda[t])*new_mu + (1 - BF.sigmoid(Qlambda[t]))*Qx_mean[t],
                             np.sqrt(dt) * driving_noise, x_names[t], learnable=True))
variational_posterior = ProbabilisticModel([Qomega] + Qx)
AR_model.set_posterior_model(variational_posterior)

# Inference #
N_itr = 500
N_smpl = 10
optimizer = "SGD"
lr = 0.001
inference.perform_inference(AR_model,
                            number_iterations=N_itr,
                            number_samples=N_smpl,
                            optimizer=optimizer,
                            lr=lr)

loss_list1 = AR_model.diagnostics["loss curve"]

# ELBO
N_ELBO_smpl = 1000
ELBO = AR_model.estimate_log_model_evidence(N_ELBO_smpl)
print("The ELBO is {}".format(ELBO))

# Statistics
posterior_samples1 = AR_model._get_posterior_sample(2000)
omega_posterior_samples1 = posterior_samples1[omega].detach().numpy().flatten()
omega_mean1 = np.mean(omega_posterior_samples1)
omega_sd1 = np.sqrt(np.var(omega_posterior_samples1))

x_mean1 = []
lower_bound1 = []
upper_bound1 = []
for xt in x:
    x_posterior_samples1 = posterior_samples1[xt].detach().numpy().flatten()
    mean1 = np.mean(x_posterior_samples1)
    sd1 = np.sqrt(np.var(x_posterior_samples1))
    x_mean1.append(mean1)
    lower_bound1.append(mean1 - sd1)
    upper_bound1.append(mean1 + sd1)
print("The estimated coefficient is: {} +- {}".format(omega_mean1, omega_mean1))

# Mean-field variational distribution #
Qomega = NormalVariable(2*np.pi*8, 1., 'omega', learnable=True)
Qx = [NormalVariable(0., 1., 'x0', learnable=True)]

for t in range(1, T):
    Qx.append(NormalVariable(0, 2., x_names[t], learnable=True))
variational_posterior = ProbabilisticModel([Qomega] + Qx)
AR_model.set_posterior_model(variational_posterior)

# Inference #
inference.perform_inference(AR_model,
                            number_iterations=N_itr,
                            number_samples=N_smpl,
                            optimizer=optimizer,
                            lr=lr)

loss_list2 = AR_model.diagnostics["loss curve"]

# ELBO
ELBO = AR_model.estimate_log_model_evidence(N_ELBO_smpl)
print("The ELBO is {}".format(ELBO))

# Statistics
posterior_samples2 = AR_model._get_posterior_sample(2000)

x_mean2 = []
lower_bound2 = []
upper_bound2 = []
for xt in x:
    x_posterior_samples2 = posterior_samples2[xt].detach().numpy().flatten()
    mean2 = np.mean(x_posterior_samples2)
    sd2 = np.sqrt(np.var(x_posterior_samples2))
    x_mean2.append(mean2)
    lower_bound2.append(mean2 - sd2)
    upper_bound2.append(mean2 + sd2)

# Multivariate normal variational distribution #
QV = MultivariateNormalVariable(loc=np.zeros((T,)),
                                covariance_matrix=2*np.identity(T),
                                learnable=True)
Qomega = NormalVariable(2*np.pi*8, 1., 'omega', learnable=True)
Qx = [NormalVariable(QV[0], 0.1, 'x0', learnable=True)]

for t in range(1, T):
    Qx.append(NormalVariable(QV[t], 0.1, x_names[t], learnable=True))
variational_posterior = ProbabilisticModel([Qomega] + Qx)
AR_model.set_posterior_model(variational_posterior)

# Inference #
inference.perform_inference(AR_model,
                            number_iterations=N_itr,
                            number_samples=N_smpl,
                            optimizer=optimizer,
                            lr=lr)

loss_list3 = AR_model.diagnostics["loss curve"]

# ELBO
ELBO = AR_model.estimate_log_model_evidence(N_ELBO_smpl)
print("The ELBO is {}".format(ELBO))

# Statistics
posterior_samples3 = AR_model._get_posterior_sample(2000)

x_mean3 = []
lower_bound3 = []
upper_bound3 = []
for xt in x:
    x_posterior_samples3 = posterior_samples3[xt].detach().numpy().flatten()
    mean3 = np.mean(x_posterior_samples3)
    sd3 = np.sqrt(np.var(x_posterior_samples3))
    x_mean3.append(mean3)
    lower_bound3.append(mean3 - sd3)
    upper_bound3.append(mean3 + sd3)

# Structured NN distribution #
hidden_size = 10
latent_size = 10
Qomega = NormalVariable(2*np.pi*8, 1., 'omega', learnable=True)
Qepsilon = NormalVariable(np.zeros((10,1)), np.ones((10,)), 'epsilon', learnable=True)
W1 = RootVariable(np.random.normal(0, 0.1, (hidden_size, latent_size)), "W1", learnable=True)
W2 = RootVariable(np.random.normal(0, 0.1, (T, hidden_size)), "W2", learnable=True)
pre_x = BF.matmul(W2, BF.sigmoid(BF.matmul(W1, Qepsilon)))
Qx = []
for t in range(0, T):
    Qx.append(NormalVariable(pre_x[t], 1., x_names[t], learnable=True))
variational_posterior = ProbabilisticModel([Qomega] + Qx)
AR_model.set_posterior_model(variational_posterior)

# Inference #
inference.perform_inference(AR_model,
                            number_iterations=N_itr,
                            number_samples=N_smpl,
                            optimizer=optimizer,
                            lr=lr)

loss_list4 = AR_model.diagnostics["loss curve"]

# ELBO
ELBO = AR_model.estimate_log_model_evidence(N_ELBO_smpl)
print("The ELBO is {}".format(ELBO))

# Statistics
posterior_samples4 = AR_model._get_posterior_sample(2000)

x_mean4 = []
lower_bound4 = []
upper_bound4 = []
for xt in x:
    x_posterior_samples4 = posterior_samples4[xt].detach().numpy().flatten()
    mean4 = np.mean(x_posterior_samples4)
    sd4 = np.sqrt(np.var(x_posterior_samples4))
    x_mean4.append(mean4)
    lower_bound4.append(mean4 - sd4)
    upper_bound4.append(mean4 + sd4)

# Two subplots, unpack the axes array immediately
f, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(range(T), x_mean1, color="b", label="PE")
ax1.plot(range(T), x_mean2, color="r", label="MF")
ax1.plot(range(T), x_mean3, color="g", label="MV")
ax1.plot(range(T), x_mean4, color="m", label="NN")
#ax1.scatter(y_range, time_series, color="k")
ax1.plot(range(T), ground_truth, color="k", ls ="--", lw=1.5)
ax1.fill_between(range(T), lower_bound1, upper_bound1, color="b", alpha=0.25)
ax1.fill_between(range(T), lower_bound2, upper_bound2, color="r", alpha=0.25)
ax1.fill_between(range(T), lower_bound3, upper_bound3, color="g", alpha=0.25)
ax1.fill_between(range(T), lower_bound4, upper_bound4, color="m", alpha=0.25)
ax1.set_title("Time series")
ax2.plot(np.array(loss_list1), color="b")
ax2.plot(np.array(loss_list2), color="r")
ax2.plot(np.array(loss_list3), color="g")
ax2.plot(np.array(loss_list4), color="m")
ax2.set_title("Convergence")
ax2.set_xlabel("Iteration")
plt.show()
