import matplotlib.pyplot as plt
import numpy as np

from brancher.variables import RootVariable, RandomVariable, ProbabilisticModel
from brancher.standard_variables import NormalVariable, LogNormalVariable, BetaVariable
from brancher import inference
import brancher.functions as BF

# Probabilistic model #
T = 200

nu = LogNormalVariable(0.3, 1., 'nu')
x0 = NormalVariable(0., 1., 'x0')
b = BetaVariable(0.5, 1.5, 'b')

x = [x0]
names = ["x0"]
for t in range(1, T):
    names.append("x{}".format(t))
    x.append(NormalVariable(b*x[t-1], nu, names[t]))
AR_model = ProbabilisticModel(x)

# Generate data #
data = AR_model._get_sample(number_samples=1)
time_series = [float(data[xt].cpu().detach().numpy()) for xt in x]
true_b = data[b].cpu().detach().numpy()
true_nu = data[nu].cpu().detach().numpy()
print("The true coefficient is: {}".format(float(true_b)))

# Observe data #
[xt.observe(data[xt][:, 0, :]) for xt in x]

# Variational distribution #
Qnu = LogNormalVariable(0.5, 1., "nu", learnable=True)
Qb = BetaVariable(0.5, 0.5, "b", learnable=True)
variational_posterior = ProbabilisticModel([Qb, Qnu])
AR_model.set_posterior_model(variational_posterior)

# Inference #
inference.perform_inference(AR_model,
                            number_iterations=200,
                            number_samples=300,
                            optimizer='Adam',
                            lr=0.05)
loss_list = AR_model.diagnostics["loss curve"]


# Statistics
posterior_samples = AR_model._get_posterior_sample(2000)
nu_posterior_samples = posterior_samples[nu].cpu().detach().numpy().flatten()
b_posterior_samples = posterior_samples[b].cpu().detach().numpy().flatten()
b_mean = np.mean(b_posterior_samples)
b_sd = np.sqrt(np.var(b_posterior_samples))
print("The estimated coefficient is: {} +- {}".format(b_mean, b_sd))

# Two subplots, unpack the axes array immediately
f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
ax1.plot(time_series)
ax1.set_title("Time series")
ax2.plot(np.array(loss_list))
ax2.set_title("Convergence")
ax2.set_xlabel("Iteration")
ax3.hist(b_posterior_samples, 25)
ax3.axvline(x=true_b, lw=2, c="r")
ax3.set_title("Posterior samples (b)")
ax3.set_xlim(0,1)
ax4.hist(nu_posterior_samples, 25)
ax4.axvline(x=true_nu, lw=2, c="r")
ax4.set_title("Posterior samples (nu)")
plt.show()