import matplotlib.pyplot as plt
import numpy as np
import seaborn as sbn
import pandas as pd

from brancher.variables import RootVariable, RandomVariable, ProbabilisticModel
from brancher.standard_variables import NormalVariable, LogNormalVariable, BetaVariable, MultivariateNormalVariable
from brancher import inference
import brancher.functions as BF

# N repetitions
N_rep = 15 #10

# Data list
condition_list = [lambda t: (t < 10 or t > 20), lambda t: (t < 0 or t > 20), lambda t: True]
condition_label = ["Bridge", "Past", "Full"]

N_itr = 200
N_smpl = 20
optimizer = "Adam"
lr = 0.05 #0.0002
N_ELBO_smpl = 1000


for cond, label in zip(condition_list, condition_label):
    ELBO1 = []
    ELBO2 = []
    ELBO3 = []
    ELBO4 = []
    for rep in range(N_rep):
        print("Repetition: {}".format(rep))
        # Probabilistic model #
        T = 30
        dt = 0.02
        driving_noise = 0.1 #0.1
        measure_noise = 1.
        s = 10.
        r = 28.
        b = 8 / 3.
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
        y_range = [t for t in range(T) if cond(t)]
        for t in range(1, T):
            x_names.append("x{}".format(t))
            h_names.append("h{}".format(t))
            z_names.append("z{}".format(t))
            new_x = x[t - 1] + dt * s * (h[t - 1] - x[t - 1])
            new_h = h[t - 1] + dt * (x[t - 1] * (r - z[t - 1]) - h[t - 1])
            new_z = z[t - 1] + dt * (x[t - 1] * h[t - 1] - b * z[t - 1])
            x.append(NormalVariable(new_x, np.sqrt(dt) * driving_noise, x_names[t]))
            h.append(NormalVariable(new_h, np.sqrt(dt) * driving_noise, h_names[t]))
            z.append(NormalVariable(new_z, np.sqrt(dt) * driving_noise, z_names[t]))
            if t in y_range:
                y_name = "y{}".format(t)
                y_names.append(y_name)
                y.append(NormalVariable(x[t], measure_noise, y_name))
        AR_model = ProbabilisticModel(x + y)

        # Generate data #
        data = AR_model._get_sample(number_samples=1)
        time_series = [float(data[yt].data) for yt in y]
        ground_truth = [float(data[xt].data) for xt in x]

        # Observe data #
        [yt.observe(data[yt][:, 0, :]) for yt in y]

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
                l = 1.  # 2
            else:
                l = 1.  # 2
            Qx_mean.append(RootVariable(0, x_names[t] + "_mean", learnable=True))
            Qxlambda.append(RootVariable(l, x_names[t] + "_lambda", learnable=True))

            Qh_mean.append(RootVariable(0, h_names[t] + "_mean", learnable=True))
            Qhlambda.append(RootVariable(l, h_names[t] + "_lambda", learnable=True))

            Qz_mean.append(RootVariable(0, z_names[t] + "_mean", learnable=True))
            Qzlambda.append(RootVariable(l, z_names[t] + "_lambda", learnable=True))

            new_x = Qx[t - 1] + dt * s * (Qh[t - 1] - Qx[t - 1])
            new_h = Qh[t - 1] + dt * (Qx[t - 1] * (r - Qz[t - 1]) - Qh[t - 1])
            new_z = Qz[t - 1] + dt * (Qx[t - 1] * Qh[t - 1] - b * Qz[t - 1])

            Qx.append(NormalVariable(BF.sigmoid(Qxlambda[t]) * new_x + (1 - BF.sigmoid(Qxlambda[t])) * Qx_mean[t],
                                     np.sqrt(dt) * driving_noise, x_names[t], learnable=True))

            Qh.append(NormalVariable(BF.sigmoid(Qhlambda[t]) * new_h + (1 - BF.sigmoid(Qhlambda[t])) * Qh_mean[t],
                                     np.sqrt(dt) * driving_noise, h_names[t], learnable=True))

            Qz.append(NormalVariable(BF.sigmoid(Qzlambda[t]) * new_z + (1 - BF.sigmoid(Qzlambda[t])) * Qz_mean[t],
                                     np.sqrt(dt) * driving_noise, z_names[t], learnable=True))

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
        ELBO1.append(float(AR_model.estimate_log_model_evidence(N_ELBO_smpl).detach().numpy()))
        print("PE {}".format(ELBO1[-1]))

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

        loss_list2 = AR_model.diagnostics["loss curve"]

        # ELBO
        ELBO2.append(float(AR_model.estimate_log_model_evidence(N_ELBO_smpl).detach().numpy()))
        print("MF {}".format(ELBO2[-1]))

        # Multivariate normal variational distribution #

        QV = MultivariateNormalVariable(loc=np.zeros((3*T,)),
                                        scale_tril=np.identity(3*T),
                                        name="V",
                                        learnable=True)
        Qx = [NormalVariable(QV[0], 0.1, 'x0', learnable=True)]
        Qh = [NormalVariable(QV[0], 0.1, 'h0', learnable=True)]
        Qz = [NormalVariable(QV[0], 0.1, 'z0', learnable=True)]

        for t in range(1, T):
            Qx.append(NormalVariable(QV[t], driving_noise, x_names[t], learnable=True))
            Qh.append(NormalVariable(QV[T + t], driving_noise, h_names[t], learnable=True))
            Qz.append(NormalVariable(QV[2*T + t], driving_noise, z_names[t], learnable=True))
        variational_posterior = ProbabilisticModel(Qx + Qh + Qz)
        AR_model.set_posterior_model(variational_posterior)

        # Inference #
        inference.perform_inference(AR_model,
                                    number_iterations=N_itr,
                                    number_samples=N_smpl,
                                    optimizer=optimizer,
                                    lr=lr)

        loss_list3 = AR_model.diagnostics["loss curve"]

        # ELBO
        ELBO3.append(float(AR_model.estimate_log_model_evidence(N_ELBO_smpl).detach().numpy()))
        print("MN {}".format(ELBO3[-1]))

        # Structured NN distribution #
        hidden_size = 3*10
        latent_size = 3*10
        Qepsilon = NormalVariable(np.zeros((hidden_size, 1)), np.ones((hidden_size,)), 'epsilon', learnable=True)
        W1 = RootVariable(np.random.normal(0, 0.1, (hidden_size, latent_size)), "W1", learnable=True)
        W2 = RootVariable(np.random.normal(0, 0.1, (3*T, hidden_size)), "W2", learnable=True)
        pre_x = BF.matmul(W2, BF.sigmoid(BF.matmul(W1, Qepsilon)))
        Qx = []
        Qh = []
        Qz = []
        for t in range(0, T):
            Qx.append(NormalVariable(pre_x[t], driving_noise, x_names[t], learnable=True))
            Qh.append(NormalVariable(pre_x[T + t], driving_noise, h_names[t], learnable=True))
            Qz.append(NormalVariable(pre_x[2*T + t], driving_noise, z_names[t], learnable=True))
        variational_posterior = ProbabilisticModel(Qx + Qh + Qz)
        AR_model.set_posterior_model(variational_posterior)

        # Inference #
        inference.perform_inference(AR_model,
                                    number_iterations=N_itr,
                                    number_samples=N_smpl,
                                    optimizer=optimizer,
                                    lr=lr)

        loss_list4 = AR_model.diagnostics["loss curve"]

        # ELBO
        ELBO4.append(float(AR_model.estimate_log_model_evidence(N_ELBO_smpl).detach().numpy()))
        print("NN {}".format(ELBO4[-1]))

        # plt.plot(loss_list1)
        # plt.plot(loss_list2)
        # plt.plot(loss_list3)
        # plt.plot(loss_list4)
        # plt.show()

    d = {'PE': ELBO1, 'MF': ELBO2, "MN": ELBO3, "NN": ELBO4}

    import pickle
    with open('{}_lorentz_results.pickle'.format(label), 'wb') as f:
        pickle.dump(d, f)

    df = pd.DataFrame(data=d)
    df.boxplot()
    plt.title(label)
    plt.ylabel("ELBO")
    plt.savefig("Lorentz " +label+".pdf")
    plt.clf()
    #plt.show()


