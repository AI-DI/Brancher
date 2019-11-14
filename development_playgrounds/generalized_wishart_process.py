import matplotlib.pyplot as plt
import numpy as np

from brancher.variables import ProbabilisticModel
from brancher.stochastic_processes import MarkovProcess
from brancher.standard_variables import NormalVariable as Normal
from brancher.standard_variables import DeterministicVariable as Deterministic
from brancher.standard_variables import BetaVariable as Beta
from brancher.standard_variables import MultivariateNormalVariable as MultivariateNormal
import brancher.functions as BF

import pickle
from scipy.stats import zscore

path = "/home/luca/CurrencyData/exchange_file.pickle"

with open(path, 'rb') as file:
    exchange_df = pickle.load(file)

data = exchange_df.values[:, 1:].astype("float32")
indices = [True if t % 20 == 0 else False for t in range(data.shape[0])]
data = data[indices,:]
data = zscore(data)
plt.plot(data)
plt.show()

## Create time series model ##
T = sum(indices)
d = 3
nu = d + 1
a = Beta(15, 1, "a")
Id = Deterministic(0.1*np.identity(d), "Id")
U0 = Normal(np.zeros((d, 1, nu)), np.ones((d, 1, nu)), "U_0")
U = MarkovProcess(U0, lambda t, u: Normal(a*u, 0.1, "U_{}".format(t)))
Yt = [MultivariateNormal(np.zeros((d,)),
                         covariance_matrix=sum([BF.matmul(ut[:, :, k], BF.transpose(ut[:, :, k], 2, 1))
                                                for k in range(nu)]) + Id,
                         name="y_{}".format(t))
      for t, ut in enumerate(U(T).temporal_variables)]
model = ProbabilisticModel(Yt)

## Observe
[yt.observe(dt) for yt, dt in zip(Yt, data)]

#sample = model.get_sample(2) #TODO: bug

## Variational model
Qa = Beta(15, 1, "a", learnable=True)
QU = [Normal(np.zeros((d, 1, nu)), np.ones((d, 1, nu)), "U_0", learnable=True)]
Qlambda = []
Qalpha = []
cov = []
for t in range(T):
    Qlambda.append(Deterministic(0.1*np.ones((d, 1, nu)), "lambda_{}".format(t), learnable=True))
    Qalpha.append(Deterministic(0.05*np.ones((d, 1, nu)), "alpha_{}".format(t), learnable=True))
    if t > 0:
        QU.append(Normal(BF.sigmoid(Qlambda[-1])*Qa*QU[-1] + (1-BF.sigmoid(Qlambda[-1]))*Qalpha[-1],
                         0.1, "U_{}".format(t), learnable=True))
    cov.append(Deterministic(sum([BF.matmul(QU[-1][:, :, k], BF.transpose(QU[-1][:, :, k], 2, 1)) for k in range(nu)]) + Id,
                             name="cov_{}".format(t)))
model.set_posterior_model(ProbabilisticModel(QU + [Qa] + cov))
model.posterior_model.get_sample(1)

# Inference #
from brancher.inference import perform_inference
N_itr = 50
N_smpl = 1
optimizer = "SGD"
lr = 0.01
perform_inference(model,
                  number_iterations=N_itr,
                  number_samples=N_smpl,
                  optimizer=optimizer,
                  lr=lr)
loss = model.diagnostics["loss curve"]
plt.plot(loss)
plt.show()
## Inference ##

## Variationa model ##

sample = model.posterior_model.get_sample(1)

Y1_mean = 0.
Y2_mean = 0.
Y3_mean = 0.
X1_mean = 0.
N = 50
for n in range(N):
    Y1 = []
    Y2 = []
    Y3 = []
    X1 = []
    sample = model.posterior_model.get_sample(1)
    for t in range(T):
        s0 = sample["cov_{}".format(t)][0][0,0,0]
        s1 = sample["cov_{}".format(t)][0][0,1,1]
        s2 = sample["cov_{}".format(t)][0][0,2,2]
        X1.append(s1)
        Y1.append(sample["cov_{}".format(t)][0][0,1,2]/np.sqrt(s1*s2))
        Y2.append(sample["cov_{}".format(t)][0][0,0,1]/np.sqrt(s0*s1))
        Y3.append(sample["cov_{}".format(t)][0][0,0,2]/np.sqrt(s0*s2))
    Y1_mean += np.array(Y1)/float(N)
    Y2_mean += np.array(Y2)/float(N)
    Y3_mean += np.array(Y3)/float(N)
    X1_mean += np.array(X1)/float(N)
plt.plot(range(T), Y1_mean)
plt.plot(range(T), Y2_mean)
plt.plot(range(T), Y3_mean)
plt.show()

plt.plot(range(T), X1_mean)
plt.show()