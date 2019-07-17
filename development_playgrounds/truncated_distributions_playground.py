import matplotlib.pyplot as plt

from brancher.variables import ProbabilisticModel
from brancher.standard_variables import NormalVariable, LogNormalVariable

from brancher.transformations import truncate_model
from brancher.visualizations import plot_density

# Normal model
mu = NormalVariable(0., 1., "mu")
x = NormalVariable(mu, 0.1, "x")
model = ProbabilisticModel([x])

# decision rule
model_statistics = lambda dic: dic[x].data
truncation_rule = lambda a: ((a > 0.5) & (a < 0.6)) | ((a > -0.6) & (a < -0.5))

# Truncated model
truncated_model = truncate_model(model, truncation_rule, model_statistics)

plot_density(truncated_model, variables=["mu", "x"], number_samples=10000)
plt.show()
