import matplotlib.pyplot as plt
import numpy as np
import seaborn as sbn
import pandas as pd
import torch

from brancher.variables import RootVariable, RandomVariable, ProbabilisticModel
from brancher.standard_variables import NormalVariable, MultivariateNormalVariable, DeterministicVariable
from brancher.transformations import TriangularLinear, Sigmoid, Bias
from brancher import inference
import brancher.functions as BF

# N repetitions
N_rep = 15 #15

# Data list
condition_list = [lambda t: (t < 10 or t > 30), lambda t: (t < 0 or t > 30), lambda t: True]
condition_label = ["Bridge", "Past", "Full"]

N_itr = 200
N_smpl = 50
optimizer = "Adam"
lr = 0.05 #0.0002#0.0002
lr_mn = 0.015
N_ELBO_smpl = 1000


for cond, label in zip(condition_list, condition_label):
    Lk1 = []
    Lk2 = []
    Lk3 = []
    Lk4 = []
    MSE1 = []
    MSE2 = []
    MSE3 = []
    MSE4 = []
    for rep in range(N_rep):
        print("Repetition: {}".format(rep))
        # Probabilistic model #
        T = 40
        dt = 0.01
        driving_noise = 0.5
        measure_noise = 0.1
        x0 = NormalVariable(0., driving_noise, 'x0')
        y0 = NormalVariable(x0, measure_noise, 'y0')

        x = [x0]
        y = [y0]
        x_names = ["x0"]
        y_names = ["y0"]
        y_range = [t for t in range(T) if cond(t)]
        for t in range(1, T):
            x_names.append("x{}".format(t))
            x.append(NormalVariable(x[t - 1], np.sqrt(dt) * driving_noise, x_names[t]))
            if t in y_range:
                y_name = "y{}".format(t)
                y_names.append(y_name)
                y.append(NormalVariable(x[t], measure_noise, y_name))
        AR_model = ProbabilisticModel(x + y)

        # Generate data #
        data = AR_model._get_sample(number_samples=1)
        time_series = [float(data[yt].data) for yt in y]
        ground_truth = [float(data[xt].data) for xt in x]
        #true_b = data[omega].data
        #print("The true coefficient is: {}".format(float(true_b)))

        # Observe data #
        [yt.observe(data[yt][:, 0, :]) for yt in y]

        # Structured variational distribution #
        Qx = [NormalVariable(0., 1., 'x0', learnable=True)]
        Qx_mean = [RootVariable(0., 'x0_mean', learnable=True)]
        Qlambda = [RootVariable(0., 'x0_lambda', learnable=True)]

        for t in range(1, T):
            if t in y_range:
                l = 0.
            else:
                l = 0.
            Qx_mean.append(RootVariable(0, x_names[t] + "_mean", learnable=True))
            Qlambda.append(RootVariable(l, x_names[t] + "_lambda", learnable=True))
            Qx.append(NormalVariable(BF.sigmoid(Qlambda[t]) * Qx[t - 1] + (1 - BF.sigmoid(Qlambda[t])) * Qx_mean[t],
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

        # ELBO
        #ELBO1.append(float(AR_model.estimate_log_model_evidence(N_ELBO_smpl).detach().numpy()))
        #print("PC {}".format(ELBO1[-1]))

        # MSE
        posterior_samples = AR_model._get_posterior_sample(2000)

        x_mean1 = []
        lower_bound1 = []
        upper_bound1 = []
        sd_mean1 = []
        avgRMSE = 0.
        for xt in x:
            x_posterior_samples = posterior_samples[xt].detach().numpy().flatten()
            mean = np.mean(x_posterior_samples)
            sd = np.sqrt(np.var(x_posterior_samples))
            x_mean1.append(mean)
            lower_bound1.append(mean - sd)
            upper_bound1.append(mean + sd)
        MSE = np.mean((np.array(ground_truth) - np.array(x_mean1)) ** 2)
        var = 0.5*(np.array(upper_bound1) - np.array(lower_bound1))**2
        Lk = np.mean(0.5 * (np.array(ground_truth) - np.array(x_mean1)) ** 2 / var + 0.5*np.log(var) + 0.5*np.log(2 * np.pi))
        print("PC MSE {}".format(MSE))
        print("PC lk {}".format(Lk))
        MSE1.append(MSE)
        Lk1.append(Lk)

        # Mean-field variational distribution #
        Qx = [NormalVariable(0., 1., 'x0', learnable=True)]

        for t in range(1, T):
            Qx.append(NormalVariable(0, 1., x_names[t], learnable=True))
        variational_posterior = ProbabilisticModel(Qx)
        AR_model.set_posterior_model(variational_posterior)

        # Inference #
        inference.perform_inference(AR_model,
                                    number_iterations=N_itr,
                                    number_samples=N_smpl,
                                    optimizer=optimizer,
                                    lr=lr)

        loss_list2 = AR_model.diagnostics["loss curve"]

        # ELBO
        #ELBO2.append(float(AR_model.estimate_log_model_evidence(N_ELBO_smpl).detach().numpy()))
        #print("MF {}".format(ELBO2[-1]))

        # MSE
        posterior_samples = AR_model._get_posterior_sample(2000)

        x_mean2 = []
        lower_bound2 = []
        upper_bound2 = []
        for xt in x:
            x_posterior_samples = posterior_samples[xt].detach().numpy().flatten()
            mean = np.mean(x_posterior_samples)
            sd = np.sqrt(np.var(x_posterior_samples))
            x_mean2.append(mean)
            lower_bound2.append(mean - sd)
            upper_bound2.append(mean + sd)
        MSE = np.mean((np.array(ground_truth) - np.array(x_mean2)) ** 2)
        var = 0.5 * (np.array(upper_bound2) - np.array(lower_bound2)) ** 2
        Lk = np.mean(
            0.5 * (np.array(ground_truth) - np.array(x_mean2)) ** 2 / var + 0.5 * np.log(var) + 0.5 * np.log(
                2 * np.pi))
        print("MF MSE {}".format(MSE))
        print("MF lk {}".format(Lk))
        MSE2.append(MSE)
        Lk2.append(Lk)

        # Multivariate normal variational distribution #

        QV = MultivariateNormalVariable(loc=np.zeros((T,)),
                                        scale_tril=0.1*np.identity(T),
                                        learnable=True)
        Qx = [DeterministicVariable(QV[0], 'x0')]

        for t in range(1, T):
            Qx.append(DeterministicVariable(QV[t], x_names[t]))
        variational_posterior = ProbabilisticModel(Qx)
        AR_model.set_posterior_model(variational_posterior)

        # Inference #
        inference.perform_inference(AR_model,
                                    number_iterations=N_itr,
                                    number_samples=N_smpl,
                                    optimizer=optimizer,
                                    lr=lr_mn)

        loss_list3 = AR_model.diagnostics["loss curve"]

        # ELBO
        #ELBO3.append(float(AR_model.estimate_log_model_evidence(N_ELBO_smpl).detach().numpy()))
        #print("MN {}".format(ELBO3[-1]))

        # MSE
        posterior_samples = AR_model._get_posterior_sample(2000)

        x_mean3 = []
        lower_bound3 = []
        upper_bound3 = []
        for xt in x:
            x_posterior_samples = posterior_samples[xt].detach().numpy().flatten()
            mean = np.mean(x_posterior_samples)
            sd = np.sqrt(np.var(x_posterior_samples))
            x_mean3.append(mean)
            lower_bound3.append(mean - sd)
            upper_bound3.append(mean + sd)
        MSE = np.mean((np.array(ground_truth) - np.array(x_mean3)) ** 2)
        var = 0.5 * (np.array(upper_bound3) - np.array(lower_bound3)) ** 2
        Lk = np.mean(
            0.5 * (np.array(ground_truth) - np.array(x_mean3)) ** 2 / var + 0.5 * np.log(var) + 0.5 * np.log(
                2 * np.pi))
        print("MN MSE {}".format(MSE))
        print("MN lk {}".format(Lk))
        MSE3.append(MSE)
        Lk3.append(Lk)

        # Structured NN distribution #
        hidden_size = 10
        latent_size = 10
        epsilon = NormalVariable(np.zeros((hidden_size, 1)), np.ones((hidden_size,)), 'epsilon', learnable=True)
        AR_model = ProbabilisticModel(x + y + [epsilon])

        Qepsilon = NormalVariable(np.zeros((hidden_size, 1)), np.ones((hidden_size,)), 'epsilon', learnable=True)
        W1 = RootVariable(np.random.normal(0, 0.1, (hidden_size, latent_size)), "W1", learnable=True)
        W2 = RootVariable(np.random.normal(0, 0.1, (T, hidden_size)), "W2", learnable=True)
        pre_x = BF.matmul(W2, BF.sigmoid(BF.matmul(W1, Qepsilon)))
        Qx = []
        for t in range(0, T):
            Qx.append(NormalVariable(pre_x[t], driving_noise, x_names[t], learnable=True))
        variational_posterior = ProbabilisticModel(Qx)
        AR_model.set_posterior_model(variational_posterior)

        # Inference #
        inference.perform_inference(AR_model,
                                    number_iterations=N_itr,
                                    number_samples=N_smpl,
                                    optimizer=optimizer,
                                    lr=lr)

        loss_list4 = AR_model.diagnostics["loss curve"]

        # ELBO
        #ELBO4.append(float(AR_model.estimate_log_model_evidence(N_ELBO_smpl).detach().numpy()))
        #print("NN {}".format(ELBO4[-1]))

        # MSE
        posterior_samples = AR_model._get_posterior_sample(2000)

        x_mean4 = []
        lower_bound4 = []
        upper_bound4 = []
        for xt in x:
            x_posterior_samples = posterior_samples[xt].detach().numpy().flatten()
            mean = np.mean(x_posterior_samples)
            sd = np.sqrt(np.var(x_posterior_samples))
            x_mean4.append(mean)
            lower_bound4.append(mean - sd)
            upper_bound4.append(mean + sd)
        MSE = np.mean((np.array(ground_truth) - np.array(x_mean4)) ** 2)
        var = 0.5 * (np.array(upper_bound4) - np.array(lower_bound4)) ** 2
        Lk = np.mean(
            0.5 * (np.array(ground_truth) - np.array(x_mean4)) ** 2 / var + 0.5 * np.log(var) + 0.5 * np.log(
                2 * np.pi))
        print("NN MSE {}".format(MSE))
        print("NN lk {}".format(Lk))
        MSE4.append(MSE)
        Lk4.append(Lk)

        # Two subplots, unpack the axes array immediately
        f, (ax1, ax2) = plt.subplots(1, 2)
        ax1.plot(range(T), x_mean1, color="b", label="PC")
        ax1.fill_between(range(T), lower_bound1, upper_bound1, color="b", alpha=0.25)
        ax1.plot(range(T), x_mean2, color="r", label="MF")
        ax1.fill_between(range(T), lower_bound2, upper_bound2, color="r", alpha=0.25)
        ax1.plot(range(T), x_mean3, color="m", label="MN")
        ax1.fill_between(range(T), lower_bound3, upper_bound3, color="m", alpha=0.25)
        ax1.plot(range(T), x_mean4, color="g", label="NN")
        ax1.fill_between(range(T), lower_bound4, upper_bound4, color="g", alpha=0.25)
        ax1.scatter(y_range, time_series, color="k")
        ax1.plot(range(T), ground_truth, color="k", ls="--", lw=1.5)
        ax1.set_title("Time series")
        ax2.plot(np.array(loss_list1), color="b")
        ax2.plot(np.array(loss_list2), color="r")
        ax2.plot(np.array(loss_list3), color="m")
        ax2.plot(np.array(loss_list4), color="g")
        ax2.set_title("Convergence")
        # ax2.set_xlabel("Iteration")
        plt.show()

    d = {'PE': {"Lk": Lk1, "MSE": MSE1}, 'ADVI (MF)': {"Lk": Lk2, "MSE": MSE2}, "ADVI (MN)": {"Lk": Lk3, "MSE": MSE3}, "NN": {"Lk": Lk4, "MSE": MSE4}}
    c = {'PE': MSE1, 'ADVI (MF)': MSE2, "ADVI (MN)": MSE3, "NN": MSE4}

    import pickle
    with open('{}_brownian_results.pickle'.format(label), 'wb') as f:
        pickle.dump(d, f)

    df = pd.DataFrame(data=c)
    df.boxplot()
    plt.title(label)
    plt.ylabel("Brownian " + label + ".pdf")
    plt.clf()
    #plt.show()


