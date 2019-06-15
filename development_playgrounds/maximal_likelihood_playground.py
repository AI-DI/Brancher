import matplotlib.pyplot as plt
import numpy as np
import chainer.links as L

from brancher.variables import RootVariable
from brancher.standard_variables import NormalVariable
from brancher.inference import maximal_likelihood
import brancher.functions as BF
from brancher.functions import BrancherFunction as bf

# Parameters
number_regressors = 1
number_observations = 15
real_weights = np.random.normal(0, 1, (number_regressors, 1))
real_sigma = 0.6
input_variable = np.random.normal(0, 1, (number_observations, number_regressors))

# ProbabilisticModel
regression_link = bf(L.Linear(number_regressors, 1))
x = RootVariable(input_variable, "x", is_observed=True)
sigma = RootVariable(0.1, "sigma", learnable=True)
y = NormalVariable(regression_link(x), BF.exp(sigma), "y")

# Observations
data = (np.matmul(x.value.data, real_weights)
        + np.random.normal(0,real_sigma,(number_observations,1)))
y.observe(data)
print(y)

# Maximal Likelihood
loss_list = maximal_likelihood(y, number_iterations=1000)

a_range = np.linspace(-2,2,40)
model_prediction = []
for a in a_range:
    x.value = a
    sigma.value = -20.
    model_prediction.append(float(y._get_sample()[y].data))

# Two subplots, unpack the axes array immediately
f, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(np.array(loss_list))
ax1.set_title("Convergence")
ax1.set_xlabel("Iteration")
ax1.set_ylabel("Negative log-likelihood")
ax2.plot(a_range, a_range*real_weights.flatten())
ax2.plot(a_range, np.array(model_prediction), c = "red")
ax2.scatter(input_variable.flatten(), data, c = "k")
ax2.set_title("ML fit")
plt.show()