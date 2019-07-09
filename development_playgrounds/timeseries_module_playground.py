import matplotlib.pyplot as plt

from brancher.stochastic_processes import MarkovProcess
from brancher.standard_variables import NormalVariable as Normal
from brancher.standard_variables import BetaVariable as Beta
from brancher.standard_variables import LogNormalVariable as LogNormal

x0 = Normal(0, 1, "x0")
sigma = LogNormal(1, 1, "sigma")
X = MarkovProcess(x0, lambda x: Normal(x, sigma, "x"))
markov_model = X(100)

temporal_sample = markov_model.get_sample(10)

print(temporal_sample)
temporal_sample.plot()
plt.show()

