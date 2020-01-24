import matplotlib.pyplot as plt
import numpy as np

from brancher.variables import RootVariable, RandomVariable, ProbabilisticModel
from brancher.standard_variables import NormalVariable, LogNormalVariable, BetaVariable, MultivariateNormalVariable
from brancher import inference
import brancher.functions as BF

# Probabilistic model #
T = 30
dt = 0.02
driving_noise = 0.5
measure_noise = 1.
s = 10.
r = 28.
b = 8/3.
x0 = NormalVariable(0., driving_noise, 'x0')
y0 = NormalVariable(x0, measure_noise, 'y0')
h0 = NormalVariable(0., driving_noise, 'h0')
z0 = NormalVariable(0., driving_noise, 'z0')

x = [x0]
h = [h0]
z = [z0]
y = [y0]
x_names = ["x0"]
h_names = ["h0"]
z_names = ["z0"]
y_names = ["y0"]
y_range = [t for t in range(T)]
for t in range(1, T):
    x_names.append("x{}".format(t))
    h_names.append("h{}".format(t))
    z_names.append("z{}".format(t))
    new_x = x[t-1] + dt*s*(h[t-1] - x[t-1])
    new_h = h[t-1] + dt*(x[t-1]*(r - z[t-1]) - h[t-1])
    new_z = z[t-1] + dt*(x[t-1]*h[t-1] - b*z[t-1])
    x.append(NormalVariable(new_x, np.sqrt(dt)*driving_noise, x_names[t]))
    h.append(NormalVariable(new_h, np.sqrt(dt) * driving_noise, h_names[t]))
    z.append(NormalVariable(new_z, np.sqrt(dt) * driving_noise, z_names[t]))
    if t in y_range:
        y_name = "y{}".format(t)
        y_names.append(y_name)
        y.append(NormalVariable(x[t], measure_noise, y_name))
AR_model = ProbabilisticModel(x + y)

# Generate data #
data = AR_model._get_sample(number_samples=1)
time_series = [float(data[yt].data) if t != 5 and t != 25 else 20. for t, yt in enumerate(y)]
ground_truth = [float(data[xt].data) for xt in x]

# Observe data #
[yt.observe(d) for yt, d in zip(y,time_series)]

#get time series
plt.plot([data[xt][:, 0, :] for xt in x])
plt.scatter(y_range, time_series, c="k")
plt.show()


# Structured variational distribution #
Qx = [NormalVariable(0., 1., 'x0', learnable=True)]
Qx_mean = [RootVariable(0., 'x0_mean', learnable=True)]
Qxlambda = [RootVariable(0.5, 'x0_lambda', learnable=True)]

Qh = [NormalVariable(0., 1., 'h0', learnable=True)]
Qh_mean = [RootVariable(0., 'h0_mean', learnable=True)]
Qhlambda = [RootVariable(0.5, 'h0_lambda', learnable=True)]

Qz = [NormalVariable(0., 1., 'z0', learnable=True)]
Qz_mean = [RootVariable(0., 'z0_mean', learnable=True)]
Qzlambda = [RootVariable(0.5, 'z0_lambda', learnable=True)]


for t in range(1, T):
    if t in y_range:
        l = 0. #0.5
    else:
        l = 0. #0.5
    Qx_mean.append(RootVariable(0, x_names[t] + "_mean", learnable=True))
    Qxlambda.append(RootVariable(l, x_names[t] + "_lambda", learnable=True))

    Qh_mean.append(RootVariable(0, h_names[t] + "_mean", learnable=True))
    Qhlambda.append(RootVariable(l, h_names[t] + "_lambda", learnable=True))

    Qz_mean.append(RootVariable(0, z_names[t] + "_mean", learnable=True))
    Qzlambda.append(RootVariable(l, z_names[t] + "_lambda", learnable=True))

    new_x = Qx[t-1] + dt*s*(Qh[t-1] - Qx[t-1])
    new_h = Qh[t-1] + dt*(Qx[t-1]*(r - Qz[t-1]) - Qh[t-1])
    new_z = Qz[t-1] + dt*(Qx[t-1]*Qh[t-1] - b*Qz[t-1])

    Qx.append(NormalVariable(BF.sigmoid(Qxlambda[t])*new_x + (1 - BF.sigmoid(Qxlambda[t]))*Qx_mean[t],
                             np.sqrt(dt) * driving_noise, x_names[t], learnable=True))

    Qh.append(NormalVariable(BF.sigmoid(Qhlambda[t]) * new_h + (1 - BF.sigmoid(Qhlambda[t])) * Qh_mean[t],
                             np.sqrt(dt) * driving_noise, h_names[t], learnable=True))

    Qz.append(NormalVariable(BF.sigmoid(Qzlambda[t]) * new_z + (1 - BF.sigmoid(Qzlambda[t])) * Qz_mean[t],
                             np.sqrt(dt) * driving_noise, z_names[t], learnable=True))

variational_posterior = ProbabilisticModel(Qx + Qh + Qz)
AR_model.set_posterior_model(variational_posterior)
#
# # Inference #
N_itr = 100 #800
N_smpl = 10
optimizer = "SGD"
lr = 0.1
inference.perform_inference(AR_model,
                            number_iterations=N_itr,
                            number_samples=N_smpl,
                            optimizer=optimizer,
                            lr=lr)

loss_list1 = AR_model.diagnostics["loss curve"]
#
# # ELBO
N_ELBO_smpl = 150
ELBO1 = AR_model.estimate_log_model_evidence(N_ELBO_smpl)
print("The ELBO is {}".format(ELBO1))

# Statistics
posterior_samples1 = AR_model._get_posterior_sample(2000)
samples1 = AR_model.posterior_model.get_sample(1000)

x_mean1 = []
lower_bound1 = []
upper_bound1 = []

sigmoid = lambda x: 1/(1 + np.exp(-x))
lambda_list = [sigmoid(float(l._get_sample(1)[l].detach().numpy())) \
               for l in Qxlambda]

for xt in x:
    #lambda_list.append(posterior_samples1[xt])
    x_posterior_samples1 = posterior_samples1[xt].detach().numpy().flatten()
    mean1 = np.mean(x_posterior_samples1)
    sd1 = np.sqrt(np.var(x_posterior_samples1))
    x_mean1.append(mean1)
    lower_bound1.append(mean1 - sd1)
    upper_bound1.append(mean1 + sd1)

# Mean field
Qx = [NormalVariable(0., 1., 'x0', learnable=True)]
Qh = [NormalVariable(0., 1., 'h0', learnable=True)]
Qz = [NormalVariable(0., 1., 'z0', learnable=True)]


for t in range(1, T):
    Qx.append(NormalVariable(0., driving_noise, x_names[t], learnable=True))
    Qh.append(NormalVariable(0., driving_noise, h_names[t], learnable=True))
    Qz.append(NormalVariable(0., driving_noise, z_names[t], learnable=True))

variational_posterior = ProbabilisticModel(Qx + Qh + Qz)
AR_model.set_posterior_model(variational_posterior)

# Inference #
inference.perform_inference(AR_model,
                            number_iterations=N_itr,
                            number_samples=N_smpl,
                            optimizer=optimizer,
                            lr=lr)

loss_list1 = AR_model.diagnostics["loss curve"]

# ELBO
N_ELBO_smpl = 100
ELBO2 = AR_model.estimate_log_model_evidence(N_ELBO_smpl)
print("The ELBO is {}".format(ELBO2))

# Statistics
posterior_samples2 = AR_model._get_posterior_sample(2000)
samples2 = AR_model.posterior_model.get_sample(1000)

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

# Densities
from brancher.visualizations import plot_multiple_samples
plot_multiple_samples([samples1, samples2], variables=["x5", "x7"], labels=["PE", "MF"])
plt.show()


# Two subplots, unpack the axes array immediately
f, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.plot(range(T), x_mean1, color="b", label="PE")
ax1.fill_between(range(T), lower_bound1, upper_bound1, color="b", alpha=0.25)
ax1.plot(range(T), x_mean2, color="r", label="MF")
ax1.fill_between(range(T), lower_bound2, upper_bound2, color="r", alpha=0.25)
ax1.scatter(y_range, time_series, color="k")
ax1.plot(range(T), ground_truth, color="k", ls ="--", lw=1.5)
ax1.set_title("Time series")
ax2.plot(np.array(loss_list1), color="b")
ax2.set_title("Convergence")
ax2.set_xlabel("Iteration")
ax3.plot(lambda_list)
#ax3.set_xlim(0, 1)
plt.show()
