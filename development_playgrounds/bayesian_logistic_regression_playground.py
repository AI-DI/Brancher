import chainer
import matplotlib.pyplot as plt
import numpy as np

from brancher.variables import RootVariable, ProbabilisticModel
from brancher.standard_variables import NormalVariable, BinomialVariable, MultivariateNormalVariable
from brancher import inference
import brancher.functions as BF

# Data
number_regressors = 2
number_samples = 50
x1_input_variable = np.random.normal(1.5, 1.5, (int(number_samples/2), number_regressors, 1))
x1_labels = 0*np.ones((int(number_samples/2), 1))
x2_input_variable = np.random.normal(-1.5, 1.5, (int(number_samples/2), number_regressors, 1))
x2_labels = 1*np.ones((int(number_samples/2),1))
input_variable = np.concatenate((x1_input_variable, x2_input_variable), axis=0)
labels = np.concatenate((x1_labels, x2_labels), axis=0)

# Probabilistic model
weights = NormalVariable(np.zeros((1, number_regressors)), 0.5*np.ones((1, number_regressors)), "weights")
x = RootVariable(input_variable, "x", is_observed=True)
logit_p = BF.matmul(weights, x)
k = BinomialVariable(1, logits=logit_p, name="k")
model = ProbabilisticModel([k])

samples = model._get_sample(300)

# Observations
k.observe(labels)

# Variational Model
Qweights = MultivariateNormalVariable(loc=np.zeros((1, number_regressors)),
                                      covariance_matrix=np.identity(number_regressors),
                                      name="weights", learnable=True)
variational_model = ProbabilisticModel([Qweights])
model.set_posterior_model(variational_model)

# Inference
inference.perform_inference(model,
                            number_iterations=3000,
                            number_samples=50,
                            optimizer='Adam',
                            lr=0.001)

loss_list = model.diagnostics["loss curve"]

# Statistics
posterior_samples = model._get_posterior_sample(1000)
weights_posterior_samples = posterior_samples[weights].detach().numpy()

# Two subplots, unpack the axes array immediately
f, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.plot(np.array(loss_list))
ax1.set_title("Convergence")
ax1.set_xlabel("Iteration")
ax2.scatter(input_variable[:, 0, 0], input_variable[:, 1, 0], c=labels.flatten())
for w in weights_posterior_samples:
    coeff = -float(w[0, 0, 0])/float(w[0, 0, 1])
    x_range = np.linspace(-2, 2, 200)
    ax2.plot(x_range, coeff*x_range, alpha=0.1)
ax2.set_xlim(-2, 2)
ax2.set_ylim(-2, 2)
ax3.scatter(weights_posterior_samples[:, 0, 0, 0], weights_posterior_samples[:, 0, 0, 1], alpha=0.5)
ax3.set_title("Posterior scatterplot")
plt.show()