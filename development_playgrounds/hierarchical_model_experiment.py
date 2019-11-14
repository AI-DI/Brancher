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

N_repetitions = 15 #15
N_scores_list = [1,5,10,15]

ELBO1_list = []
ELBO2_list = []
ELBO3_list = []
ELBO4_list = []
ELBO5_list = []

for N_scores in N_scores_list:
    ELBO1 = []
    ELBO2 = []
    ELBO3 = []
    ELBO4 = []
    ELBO5 = []
    for rep in range(N_repetitions):
        print(rep)
        # Probabilistic model
        N_groups = 1
        N_people = 5
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
        ELBO2 += [model.estimate_log_model_evidence(1000).detach().numpy()]

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
        ELBO3 += [model.estimate_log_model_evidence(1000).detach().numpy()]

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
        ELBO4 += [model.estimate_log_model_evidence(1000).detach().numpy()]

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
        ELBO5 += [model.estimate_log_model_evidence(1000).detach().numpy()]

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
        ELBO1 += [model.estimate_log_model_evidence(1000).detach().numpy()]
    ELBO1_list += [ELBO1]
    ELBO2_list += [ELBO2]
    ELBO3_list += [ELBO3]
    ELBO4_list += [ELBO4]
    ELBO5_list += [ELBO5]

ELBO1_mean = [np.mean(l) for l in ELBO1_list]
ELBO2_mean = [np.mean(l) for l in ELBO2_list]
ELBO3_mean = [np.mean(l) for l in ELBO3_list]
ELBO4_mean = [np.mean(l) for l in ELBO4_list]
ELBO5_mean = [np.mean(l) for l in ELBO5_list]

ELBO1_sd = [np.std(l)/np.sqrt(N_repetitions) for l in ELBO1_list]
ELBO2_sd = [np.std(l)/np.sqrt(N_repetitions) for l in ELBO2_list]
ELBO3_sd = [np.std(l)/np.sqrt(N_repetitions) for l in ELBO3_list]
ELBO4_sd = [np.std(l)/np.sqrt(N_repetitions) for l in ELBO4_list]
ELBO5_sd = [np.std(l)/np.sqrt(N_repetitions) for l in ELBO5_list]

lower_bound1 = np.array(ELBO1_mean) - np.array(ELBO1_sd)
lower_bound2 = np.array(ELBO2_mean) - np.array(ELBO2_sd)
lower_bound3 = np.array(ELBO3_mean) - np.array(ELBO3_sd)
lower_bound4 = np.array(ELBO4_mean) - np.array(ELBO4_sd)
lower_bound5 = np.array(ELBO5_mean) - np.array(ELBO5_sd)

upper_bound1 = np.array(ELBO1_mean) + np.array(ELBO1_sd)
upper_bound2 = np.array(ELBO2_mean) + np.array(ELBO2_sd)
upper_bound3 = np.array(ELBO3_mean) + np.array(ELBO3_sd)
upper_bound4 = np.array(ELBO4_mean) + np.array(ELBO4_sd)
upper_bound5 = np.array(ELBO5_mean) + np.array(ELBO5_sd)

plt.plot(N_scores_list, ELBO1_mean, label="Structured (PE)", color="b")
plt.fill_between(N_scores_list,lower_bound1,upper_bound1, alpha=0.5, color="b")
plt.plot(N_scores_list, ELBO3_mean, label="Structured (NN)", color="r")
plt.fill_between(N_scores_list,lower_bound3,upper_bound3, alpha=0.5, color="r")
plt.plot(N_scores_list, ELBO4_mean, label="Structured (Rank {} normal)".format(rank), color="g")
plt.fill_between(N_scores_list,lower_bound4,upper_bound4, alpha=0.5, color="g")
plt.plot(N_scores_list, ELBO5_mean, label="Structured (Full rank)".format(rank), color="c")
plt.fill_between(N_scores_list,lower_bound5,upper_bound5, alpha=0.5, color="c")
plt.plot(N_scores_list, ELBO2_mean, label="MF", color="m")
plt.fill_between(N_scores_list,lower_bound2,upper_bound2, alpha=0.5, color="m")
plt.legend(loc="best")
plt.xlabel("N observations")
plt.ylabel("ELBO")
plt.xlim(1,15)
plt.savefig("HirExp.pdf")
plt.show()
#
# plt.bar([0,1,2,3,4], [ELBO1, ELBO3, ELBO4, ELBO5, ELBO2], tick_label=["PE", "NN", "MN {}".format(rank), "MN full", "MF"])
# plt.show()

# Plot posterior
#plot_density(model.posterior_model, variables=["group_mean_0", "person_0", "person_1"])
#plt.show()


