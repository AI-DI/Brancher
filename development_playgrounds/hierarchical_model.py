import torch.nn as nn
import torchvision

import matplotlib.pyplot as plt
import numpy as np

from brancher.variables import RootVariable, ProbabilisticModel
from brancher.standard_variables import NormalVariable as Normal
from brancher.standard_variables import MultivariateNormalVariable as MultivariateNormal
from brancher.standard_variables import DeterministicVariable as Deterministic
from brancher import inference
import brancher.functions as BF

from brancher.visualizations import plot_density

# Probabilistic model
N_groups = 1
N_people = 5
N_scores = 10 # 5
group_means = [Normal(0., 4., "group_mean_{}".format(n)) for n in range(N_groups)]
assignment_matrix = [[1], [1], [1], [1], [1]]
people_means = [Normal(sum([l*m for l, m in zip(assignment_list, group_means)]), 0.1, "person_{}".format(m)) for m, assignment_list in enumerate(assignment_matrix)]
scores = [Normal(people_means[m], 0.1, "score_{}_{}".format(m, z)) for m in range(N_people) for z in range(N_scores)]
model = ProbabilisticModel(scores)

# Observations
sample = model.get_sample(1)
data = sample.filter(regex="^score").filter(regex="^((?!scale).)*$")
model.observe(data)

# Variational model
Qgroup_means = [Normal(0., 4., "group_mean_{}".format(n), learnable=True) for n in range(N_groups)]
Qpeople_means = [Normal(0., 0.1, "person_{}".format(m), learnable=True)
                 for m, assignment_list in enumerate(assignment_matrix)]
model.set_posterior_model(ProbabilisticModel(Qpeople_means + Qgroup_means))

# Inference #
N_itr = 300
N_smpl = 50
optimizer = "SGD"
lr = 0.00001
inference.perform_inference(model,
                            number_iterations=N_itr,
                            number_samples=N_smpl,
                            optimizer=optimizer,
                            lr=lr)
loss_list2 = model.diagnostics["loss curve"]

N_ELBO = 1000
ELBO2 = model.estimate_log_model_evidence(1000)

# Structured NN distribution #
hidden_size = 5
latent_size = 5
out_size = N_groups + N_people
Qepsilon = Normal(np.zeros((latent_size, 1)), np.ones((latent_size,)), 'epsilon', learnable=True)
W1 = RootVariable(np.random.normal(0, 0.1, (hidden_size, latent_size)), "W1", learnable=True)
W2 = RootVariable(np.random.normal(0, 0.1, (out_size, hidden_size)), "W2", learnable=True)
pre_x = BF.matmul(W2, BF.sigmoid(BF.matmul(W1, Qepsilon)))

Qgroup_means = [Normal(pre_x[n], 4., "group_mean_{}".format(n), learnable=True) for n in range(N_groups)]
Qpeople_means = [Normal(pre_x[N_groups + m], 0.1, "person_{}".format(m), learnable=True)
                 for m, assignment_list in enumerate(assignment_matrix)]

model.set_posterior_model(ProbabilisticModel(Qpeople_means + Qgroup_means))

# Inference #
inference.perform_inference(model,
                            number_iterations=N_itr,
                            number_samples=N_smpl,
                            optimizer=optimizer,
                            lr=lr)
loss_list3 = model.diagnostics["loss curve"]
ELBO3 = model.estimate_log_model_evidence(1000)

# Structured Gaussian distribution (low rank) #
rank = 1
cov_factor = RootVariable(np.random.normal(0, 0.1, (out_size, rank)), "cov_factor")
cov_shift = RootVariable(0.01*np.identity(out_size), "cov_shift", learnable=False)
mean_shift = RootVariable(np.zeros((out_size,)), "mean_shift", learnable=True)
QV = MultivariateNormal(loc=mean_shift,
                        covariance_matrix=cov_shift + BF.matmul(cov_factor, BF.transpose(cov_factor, 2, 1)),
                        name="V",
                        learnable=True)

Qgroup_means = [Normal(QV[n], 4., "group_mean_{}".format(n), learnable=True) for n in range(N_groups)]
Qpeople_means = [Normal(QV[N_groups + m], 0.1, "person_{}".format(m), learnable=True)
                 for m, assignment_list in enumerate(assignment_matrix)]

model.set_posterior_model(ProbabilisticModel(Qpeople_means + Qgroup_means))

# Inference #
inference.perform_inference(model,
                            number_iterations=N_itr,
                            number_samples=N_smpl,
                            optimizer=optimizer,
                            lr=lr)
loss_list4 = model.diagnostics["loss curve"]
ELBO4 = model.estimate_log_model_evidence(1000)

# Structured Gaussian distribution (full rank) #
cov_factor = RootVariable(np.random.normal(0, 0.1, (out_size, out_size)), "cov_factor")
cov_shift = RootVariable(0.01*np.identity(out_size), "cov_shift", learnable=False)
mean_shift = RootVariable(np.zeros((out_size,)), "mean_shift", learnable=True)
QV = MultivariateNormal(loc=mean_shift,
                        covariance_matrix=cov_shift + BF.matmul(cov_factor, BF.transpose(cov_factor, 2, 1)),
                        name="V",
                        learnable=True)

Qgroup_means = [Normal(QV[n], 4., "group_mean_{}".format(n), learnable=True) for n in range(N_groups)]
Qpeople_means = [Normal(QV[N_groups + m], 0.1, "person_{}".format(m), learnable=True)
                 for m, assignment_list in enumerate(assignment_matrix)]

model.set_posterior_model(ProbabilisticModel(Qpeople_means + Qgroup_means))

# Inference #
inference.perform_inference(model,
                            number_iterations=N_itr,
                            number_samples=N_smpl,
                            optimizer=optimizer,
                            lr=lr)
loss_list5 = model.diagnostics["loss curve"]
ELBO5 = model.estimate_log_model_evidence(1000)

# Variational model
Qgroup_means = [Normal(0., 4., "group_mean_{}".format(n), learnable=True) for n in range(N_groups)]

Qlambda = [Deterministic(2., "lambda_{}".format(m), learnable=True) for m in range(N_people)]
Qalpha = [Deterministic(0., "alpha_{}".format(m), learnable=True) for m in range(N_people)]
Qpeople_means = [Normal(BF.sigmoid(Qlambda[m])*sum([l*m for l, m in zip(assignment_list, Qgroup_means)]) + (1 - BF.sigmoid(Qlambda[m]))*Qalpha[m],
                       0.1, "person_{}".format(m), learnable=True)
                 for m, assignment_list in enumerate(assignment_matrix)]
model.set_posterior_model(ProbabilisticModel(Qpeople_means))

# Inference #
inference.perform_inference(model,
                            number_iterations=N_itr,
                            number_samples=N_smpl,
                            optimizer=optimizer,
                            lr=lr)
loss_list1 = model.diagnostics["loss curve"]
ELBO1 = model.estimate_log_model_evidence(1000)

plt.plot(loss_list1, label="Structured (PE)")
plt.plot(loss_list3, label="Structured (NN)")
plt.plot(loss_list4, label="Structured (Rank {} normal)".format(rank))
plt.plot(loss_list5, label="Structured (Full rank)".format(rank))
plt.plot(loss_list2, label="MF")
plt.legend(loc="best")
plt.show()

plt.bar([0,1,2,3,4], [ELBO1, ELBO3, ELBO4, ELBO5, ELBO2], tick_label=["PE", "NN", "MN {}".format(rank), "MN full", "MF"])
plt.show()

# Plot posterior
#plot_density(model.posterior_model, variables=["group_mean_0", "person_0", "person_1"])
#plt.show()


