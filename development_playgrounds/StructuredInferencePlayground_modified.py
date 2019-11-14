import chainer
import matplotlib.pyplot as plt
import numpy as np

from brancher.variables import RootVariable, RandomVariable, ProbabilisticModel
from brancher.standard_variables import NormalVariable, LogNormalVariable, BetaVariable, MultivariateNormalVariable
from brancher import inference
import brancher.functions as BF

# Probabilistic model #
T = 50
driving_noise = 0.1
measure_noise = 0.1
x0 = NormalVariable(-2., driving_noise, 'x0')
y0 = NormalVariable(x0, measure_noise, 'y0')
b = BetaVariable(8., 1., 'b')
drift = 0.02

x = [x0]
y = [y0]
x_names = ["x0"]
y_names = ["y0"]
y_range = [t for t in range(T) if (t < 20 or t > 30)]
for t in range(1, T):
    x_names.append("x{}".format(t))
    x.append(NormalVariable(b * (x[t - 1] - drift*(t-1)) + drift*t, driving_noise, x_names[t]))
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
Qb = BetaVariable(8., 1., "b", learnable=True)
Qx = [NormalVariable(0., 1., 'x0', learnable=True)]
Qx_mean = [RootVariable(0., 'x0_mean', learnable=True)]
Qlambda = [RootVariable(-1., 'x0_lambda', learnable=True)]


for t in range(1, T):
    if t in y_range:
        l = -1.
    else:
        l = 1.
    Qx_mean.append(RootVariable(0, x_names[t] + "_mean", learnable=True))
    Qlambda.append(RootVariable(l, x_names[t] + "_lambda", learnable=True))
    Qx.append(NormalVariable(BF.sigmoid(Qlambda[t])*(Qb*(Qx[t - 1] - drift*(t-1)) + drift*t) + (1 - BF.sigmoid(Qlambda[t]))*Qx_mean[t],
                             driving_noise, x_names[t], learnable=True))
variational_posterior = ProbabilisticModel([Qb] + Qx)
AR_model.set_posterior_model(variational_posterior)

# Inference #
N_itr = 400
N_smpl = 15
optimizer = "SGD"
lr = 0.0002
inference.perform_inference(AR_model,
                            number_iterations=N_itr,
                            number_samples=N_smpl,
                            optimizer=optimizer,
                            lr=lr)

loss_list1 = AR_model.diagnostics["loss curve"]

# ELBO
N_ELBO_smpl = 15000
ELBO = AR_model.estimate_log_model_evidence(N_ELBO_smpl)
print("The ELBO is {}".format(ELBO))

# Statistics
posterior_samples1 = AR_model._get_posterior_sample(2000)
b_posterior_samples1 = posterior_samples1[b].detach().numpy().flatten()
b_mean1 = np.mean(b_posterior_samples1)
b_sd1 = np.sqrt(np.var(b_posterior_samples1))

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
print("The estimated coefficient is: {} +- {}".format(b_mean1, b_sd1))

# # Two subplots, unpack the axes array immediately
# f, (ax1, ax2, ax3) = plt.subplots(1, 3)
# ax1.plot(range(T), x_mean1, color="b", label="PE")
# ax1.scatter(y_range, time_series, color="k")
# ax1.plot(range(T), ground_truth, color="k", ls ="--", lw=1.5)
# ax1.fill_between(range(T), lower_bound1, upper_bound1, color="b", alpha=0.25)
# ax1.set_title("Time series")
# ax2.plot(np.array(loss_list1), color="b")
# ax2.set_title("Convergence")
# ax2.set_xlabel("Iteration")
# ax3.hist(b_posterior_samples1, 25, color="b", alpha=0.25)
# ax3.set_title("Posterior samples (b)")
# ax3.set_xlim(0, 1)
# plt.show()

# Mean-field variational distribution #
Qb = BetaVariable(8., 1., "b", learnable=True)
Qx = [NormalVariable(0., 1., 'x0', learnable=True)]

for t in range(1, T):
    Qx.append(NormalVariable(0, 2., x_names[t], learnable=True))
variational_posterior = ProbabilisticModel([Qb] + Qx)
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
b_posterior_samples2 = posterior_samples2[b].detach().numpy().flatten()
b_mean2 = np.mean(b_posterior_samples2)
b_sd2 = np.sqrt(np.var(b_posterior_samples2))

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
print("The estimated coefficient is: {} +- {}".format(b_mean2, b_sd2))

# Multivariate normal variational distribution #
QV = MultivariateNormalVariable(loc=np.zeros((T,)),
                                covariance_matrix=2*np.identity(T),
                                learnable=True)
Qb = BetaVariable(8., 1., "b", learnable=True)
Qx = [NormalVariable(QV[0], 0.1, 'x0', learnable=True)]

for t in range(1, T):
    Qx.append(NormalVariable(QV[t], 0.1, x_names[t], learnable=True))
variational_posterior = ProbabilisticModel([Qb] + Qx)
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
b_posterior_samples3 = posterior_samples3[b].detach().numpy().flatten()
b_mean3 = np.mean(b_posterior_samples3)
b_sd3 = np.sqrt(np.var(b_posterior_samples3))

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
print("The estimated coefficient is: {} +- {}".format(b_mean3, b_sd3))

# Structured NN distribution #
latent_size = 10
hidden_size = 10
Qb = BetaVariable(8., 1., "b", learnable=True)
Qepsilon = NormalVariable(np.zeros((10,1)), np.ones((10,)), 'epsilon', learnable=True)
W1 = RootVariable(np.random.normal(0, 0.1, (hidden_size, latent_size)), "W1", learnable=True)
W2 = RootVariable(np.random.normal(0, 0.1, (T, hidden_size)), "W2", learnable=True)
pre_x = BF.matmul(W2, BF.sigmoid(BF.matmul(W1, Qepsilon)))
Qx = []
for t in range(0, T):
    Qx.append(NormalVariable(pre_x[t], 1., x_names[t], learnable=True))
variational_posterior = ProbabilisticModel([Qb] + Qx)
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
b_posterior_samples4 = posterior_samples4[b].detach().numpy().flatten()
b_mean4 = np.mean(b_posterior_samples4)
b_sd4 = np.sqrt(np.var(b_posterior_samples2))

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
print("The estimated coefficient is: {} +- {}".format(b_mean4, b_sd4))

# Two subplots, unpack the axes array immediately
f, (ax1, ax2, ax3) = plt.subplots(1, 3)
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
ax3.hist(b_posterior_samples1, 25, color="b", alpha=0.25)
ax3.hist(b_posterior_samples2, 25, color="r", alpha=0.25)
ax3.hist(b_posterior_samples3, 25, color="g", alpha=0.25)
ax3.hist(b_posterior_samples4, 25, color="m", alpha=0.25)
ax3.axvline(x=true_b, lw=2, c="r")
ax3.set_title("Posterior samples (b)")
ax3.set_xlim(0, 1)
plt.show()


