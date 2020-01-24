import matplotlib.pyplot as plt
import numpy as np

from brancher.variables import RootVariable, RandomVariable, ProbabilisticModel
from brancher.standard_variables import NormalVariable, DeterministicVariable
from brancher import inference
import brancher.functions as BF


## c02 data ##
from sklearn.datasets import fetch_openml

def load_mauna_loa_atmospheric_co2():
    ml_data = fetch_openml(data_id=41187)
    months = []
    ppmv_sums = []
    counts = []

    y = ml_data.data[:, 0]
    m = ml_data.data[:, 1]
    month_float = y + (m - 1) / 12
    ppmvs = ml_data.target

    for month, ppmv in zip(month_float, ppmvs):
        if not months or month != months[-1]:
            months.append(month)
            ppmv_sums.append(ppmv)
            counts.append(1)
        else:
            # aggregate monthly sum to produce average
            ppmv_sums[-1] += ppmv
            counts[-1] += 1

    months = np.asarray(months).reshape(-1, 1)
    avg_ppmvs = np.asarray(ppmv_sums) / counts
    return months, avg_ppmvs


X, y = load_mauna_loa_atmospheric_co2()

import scipy.signal as sg

T = 50
short_y = y[:T]
short_y = sg.detrend(short_y, type='linear')
noise = 0.25 #0.5
noisy_y = short_y + np.random.normal(0, noise, (T,))
#plt.plot(short_y)
#plt.scatter(range(T), noisy_y)
#plt.show()
## c02 data ##

N_itr = 500 #500 #250
N_smpl = 50
optimizer = "Adam"
lr = 0.1 #0.0002

# Probabilistic model #
transform = lambda x: x #+ 0.5*x**5
dt = 0.01
driving_noise = 0.5
measure_noise = noise
x0 = NormalVariable(0., driving_noise, 'x0')
y0 = NormalVariable(x0, measure_noise, 'y0')
x1 = NormalVariable(0., driving_noise, 'x1')
y1 = NormalVariable(x1, measure_noise, 'y1')
b = 35
omega = NormalVariable(2*np.pi*7.5, 1., "omega")

x = [x0, x1]
y = [y0, y1]
x_names = ["x0", "x1"]
y_names = ["y0", "y1"]
y_range = [t for t in range(T) if (t < 15 or t > T - 15)]
for t in range(2, T):
    x_names.append("x{}".format(t))
    #new_mu = (-1 - omega**2*dt**2 + b*dt)*x[t - 2] + (2 - b*dt)*x[t - 1]
    new_mu = (-1 + b * dt) * x[t - 2] - omega ** 2 * dt ** 2*(BF.sin(x[t - 2])) + (2 - b * dt) * x[t - 1]
    x.append(NormalVariable(new_mu, np.sqrt(dt)*driving_noise, x_names[t]))
    if t in y_range:
        y_name = "y{}".format(t)
        y_names.append(y_name)
        y.append(NormalVariable(transform(x[t]), measure_noise, y_name))
AR_model = ProbabilisticModel(x + y)

# Generate data #
data = AR_model._get_sample(number_samples=1)
time_series = [float(data[xt].data) for xt in x]
plt.plot(time_series)
plt.show()
ground_truth = short_y
#true_b = data[omega].data
#print("The true coefficient is: {}".format(float(true_b)))

# Observe data #
[yt.observe(noisy_y[t]) for t, yt in zip(y_range, y)]


# Structured variational distribution #
Qomega = NormalVariable(2*np.pi*7.5, 1., "omega", learnable=True)
Qx = [NormalVariable(0., 0.1, 'x0', learnable=True),
      NormalVariable(0., 0.1, 'x1', learnable=True)]
Qx_mean = [RootVariable(0., 'x0_mean', learnable=True),
           RootVariable(0., 'x1_mean', learnable=True)]
Qlambda = [RootVariable(0., 'x0_lambda', learnable=True),
           RootVariable(0., 'x1_lambda', learnable=True)]


for t in range(2, T):
    if t in y_range:
        l = 0.
    else:
        l = 0.
    Qx_mean.append(RootVariable(0, x_names[t] + "_mean", learnable=True))
    Qlambda.append(RootVariable(l, x_names[t] + "_lambda", learnable=True))
    new_mu = (-1 + b * dt) * Qx[t - 2] - Qomega ** 2 * dt ** 2*(BF.sin(Qx[t - 2])) + (2 - b * dt) * Qx[t - 1]
    #new_mu = (-1 - Qomega ** 2 * dt ** 2 + b * dt) * Qx[t - 2] + (2 - b * dt) * Qx[t - 1]
    Qx.append(NormalVariable(BF.sigmoid(Qlambda[t])*new_mu + (1 - BF.sigmoid(Qlambda[t]))*Qx_mean[t],
                             driving_noise, x_names[t], learnable=True))
variational_posterior = ProbabilisticModel(Qx)
AR_model.set_posterior_model(variational_posterior)

# Inference #
inference.perform_inference(AR_model,
                            number_iterations=N_itr,
                            number_samples=N_smpl,
                            optimizer=optimizer,
                            lr=lr)

loss_list1 = AR_model.diagnostics["loss curve"]

N_ELBO_smpl = 1000
ELBO1 = AR_model.estimate_log_model_evidence(N_ELBO_smpl)
print("PE {}".format(ELBO1))

# Statistics
posterior_samples1 = AR_model._get_posterior_sample(2000)

x_mean1 = []
lower_bound1 = []
upper_bound1 = []

sigmoid = lambda x: 1/(1 + np.exp(-x))
lambda_list = [sigmoid(float(l._get_sample(1)[l].detach().numpy())) \
               for l in Qlambda]

for xt in x:
    x_posterior_samples1 = transform(posterior_samples1[xt].detach().numpy().flatten())
    mean1 = np.mean(x_posterior_samples1)
    sd1 = np.sqrt(np.var(x_posterior_samples1))
    x_mean1.append(mean1)
    lower_bound1.append(mean1 - sd1)
    upper_bound1.append(mean1 + sd1)

# Mean-field variational distribution #
Qomega = NormalVariable(2*np.pi*7.5, 1., "omega", learnable=True)
Qx = [NormalVariable(0., 1., 'x0', learnable=True)]

for t in range(1, T):
    Qx.append(NormalVariable(0, 0.1, x_names[t], learnable=True))
variational_posterior = ProbabilisticModel(Qx + [Qomega])
AR_model.set_posterior_model(variational_posterior)

# Inference #
inference.perform_inference(AR_model,
                            number_iterations=N_itr,
                            number_samples=N_smpl,
                            optimizer=optimizer,
                            lr=lr)

loss_list2 = AR_model.diagnostics["loss curve"]

# ELBO
ELBO2 = AR_model.estimate_log_model_evidence(N_ELBO_smpl)
print("MF {}".format(ELBO2))

# Statistics
posterior_samples2 = AR_model._get_posterior_sample(2000)

x_mean2 = []
lower_bound2 = []
upper_bound2 = []
for xt in x:
    x_posterior_samples2 = transform(posterior_samples2[xt].detach().numpy().flatten())
    mean2 = np.mean(x_posterior_samples2)
    sd2 = np.sqrt(np.var(x_posterior_samples2))
    x_mean2.append(mean2)
    lower_bound2.append(mean2 - sd2)
    upper_bound2.append(mean2 + sd2)
#
# # Multivariate normal variational distribution #
# QV = MultivariateNormalVariable(loc=np.zeros((T,)),
#                                 covariance_matrix=2*np.identity(T),
#                                 learnable=True)
# Qx = [NormalVariable(QV[0], 0.1, 'x0', learnable=True)]
#
# for t in range(1, T):
#     Qx.append(NormalVariable(QV[t], 0.1, x_names[t], learnable=True))
# variational_posterior = ProbabilisticModel(Qx)
# AR_model.set_posterior_model(variational_posterior)
#
# # Inference #
# inference.perform_inference(AR_model,
#                             number_iterations=N_itr,
#                             number_samples=N_smpl,
#                             optimizer=optimizer,
#                             lr=lr)
#
# loss_list3 = AR_model.diagnostics["loss curve"]
#
# # ELBO
# ELBO3 = AR_model.estimate_log_model_evidence(N_ELBO_smpl)
# print("MN {}".format(ELBO3))
#
# # Statistics
# posterior_samples3 = AR_model._get_posterior_sample(2000)
#
# x_mean3 = []
# lower_bound3 = []
# upper_bound3 = []
# for xt in x:
#     x_posterior_samples3 = posterior_samples3[xt].detach().numpy().flatten()
#     mean3 = np.mean(x_posterior_samples3)
#     sd3 = np.sqrt(np.var(x_posterior_samples3))
#     x_mean3.append(mean3)
#     lower_bound3.append(mean3 - sd3)
#     upper_bound3.append(mean3 + sd3)
#
# # Structured NN distribution #
# hidden_size = 10
# latent_size = 10
# Qepsilon = NormalVariable(np.zeros((10,1)), np.ones((10,)), 'epsilon', learnable=True)
# W1 = RootVariable(np.random.normal(0, 0.1, (hidden_size, latent_size)), "W1", learnable=True)
# W2 = RootVariable(np.random.normal(0, 0.1, (T, hidden_size)), "W2", learnable=True)
# pre_x = BF.matmul(W2, BF.sigmoid(BF.matmul(W1, Qepsilon)))
# Qx = []
# for t in range(0, T):
#     Qx.append(NormalVariable(pre_x[t], 1., x_names[t], learnable=True))
# variational_posterior = ProbabilisticModel(Qx)
# AR_model.set_posterior_model(variational_posterior)
#
# # Inference #
# inference.perform_inference(AR_model,
#                             number_iterations=N_itr,
#                             number_samples=N_smpl,
#                             optimizer=optimizer,
#                             lr=lr)
#
# loss_list4 = AR_model.diagnostics["loss curve"]
#
# # ELBO
# ELBO4 = AR_model.estimate_log_model_evidence(N_ELBO_smpl)
# print("NN {}".format(ELBO4))
#
# # Statistics
# posterior_samples4 = AR_model._get_posterior_sample(2000)
#
# x_mean4 = []
# lower_bound4 = []
# upper_bound4 = []
# for xt in x:
#     x_posterior_samples4 = posterior_samples4[xt].detach().numpy().flatten()
#     mean4 = np.mean(x_posterior_samples4)
#     sd4 = np.sqrt(np.var(x_posterior_samples4))
#     x_mean4.append(mean4)
#     lower_bound4.append(mean4 - sd4)
#     upper_bound4.append(mean4 + sd4)
#
# Two subplots, unpack the axes array immediately
f, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.plot(range(T), x_mean1, color="b", label="PE")
ax1.plot(range(T), x_mean2, color="r", label="MF")
#ax1.plot(range(T), x_mean3, color="g", label="MV")
#ax1.plot(range(T), x_mean4, color="m", label="NN")
#ax1.scatter(y_range, time_series, color="k")
ax1.plot(range(T), ground_truth, color="k", ls ="--", lw=1.5)
ax1.scatter(y_range, [noisy_y[t] for t in y_range], c="k")
ax1.fill_between(range(T), lower_bound1, upper_bound1, color="b", alpha=0.25)
ax1.fill_between(range(T), lower_bound2, upper_bound2, color="r", alpha=0.25)
#ax1.fill_between(range(T), lower_bound3, upper_bound3, color="g", alpha=0.25)
#ax1.fill_between(range(T), lower_bound4, upper_bound4, color="m", alpha=0.25)
ax1.set_title("Time series")
ax2.plot(np.array(loss_list1), color="b")
ax2.plot(np.array(loss_list2), color="r")
#ax2.plot(np.array(loss_list3), color="g")
#ax2.plot(np.array(loss_list4), color="m")
ax2.set_title("Convergence")
ax2.set_xlabel("Iteration")
ax3.plot(lambda_list)
#ax3.set_xlim(0, 1)
ax3.set_title("Lambda")
plt.show()

plt.show()
