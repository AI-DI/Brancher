import matplotlib.pyplot as plt
import numpy as np

from brancher.variables import RootVariable, ProbabilisticModel
from brancher.standard_variables import NormalVariable, BinomialVariable, EmpiricalVariable, RandomIndices
from brancher import inference
import brancher.functions as BF

from brancher.config import device
print('Device used: ' + device.type)

# Data #TODO: Implement minibatch correction
number_regressors = 2
dataset_size = 50
x1_input_variable = np.random.normal(1.5, 1.5, (int(dataset_size/2), number_regressors, 1))
x1_labels = 0*np.ones((int(dataset_size/2), 1))
x2_input_variable = np.random.normal(-1.5, 1.5, (int(dataset_size/2), number_regressors, 1))
x2_labels = 1*np.ones((int(dataset_size/2),1))
input_variable = np.concatenate((x1_input_variable, x2_input_variable), axis=0)
output_labels = np.concatenate((x1_labels, x2_labels), axis=0)

# Probabilistic model
minibatch_size = 30
minibatch_indices = RandomIndices(dataset_size=dataset_size, batch_size=minibatch_size, name="indices", is_observed=True)
x = EmpiricalVariable(input_variable, indices=minibatch_indices, name="x", is_observed=True)
labels = EmpiricalVariable(output_labels, indices=minibatch_indices, name="labels", is_observed=True)
weights = NormalVariable(np.zeros((1, number_regressors)), 0.5*np.ones((1, number_regressors)), "weights")
logit_p = BF.matmul(weights, x)
k = BinomialVariable(1, logits=logit_p, name="k")
model = ProbabilisticModel([k])

#samples = model._get_sample(300)
#model.calculate_log_probability(samples)

# Observations
k.observe(labels)

#observed_model = inference.get_observed_model(model)
#observed_samples = observed_model._get_sample(number_samples=1, observed=True)

# Variational Model
Qweights = NormalVariable(np.zeros((1, number_regressors)),
                          np.ones((1, number_regressors)), "weights", learnable=True)
model.set_posterior_model(ProbabilisticModel([Qweights]))

# Inference
inference.perform_inference(model,
                            number_iterations=200,
                            number_samples=100,
                            optimizer='Adam',
                            lr=0.05)
loss_list = model.diagnostics["loss curve"]
plt.plot(loss_list)
plt.show()

# Statistics
posterior_samples = model._get_posterior_sample(50)
weights_posterior_samples = posterior_samples[weights].cpu().detach().numpy()

# Two subplots, unpack the axes array immediately
f, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(np.array(loss_list))
ax1.set_title("Convergence")
ax1.set_xlabel("Iteration")
x_range = np.linspace(-2,2,200)
ax2.scatter(input_variable[:, 0, 0], input_variable[:, 1, 0], c=output_labels.flatten())
for w in weights_posterior_samples:
    coeff = -float(w[0, 0, 0])/float(w[0, 0, 1])
    plt.plot(x_range, coeff*x_range, alpha=0.3)
ax2.set_xlim(-2,2)
ax2.set_ylim(-2,2)
plt.show()