import numpy as np
import matplotlib.pyplot as plt

from brancher.variables import ProbabilisticModel
from brancher.standard_variables import MultivariateNormalVariable

mean = np.zeros((2, 1))
covariance_matrix = np.array([[1., -0.3],
                              [-0.3, 1.]])

x = MultivariateNormalVariable(mean, covariance_matrix=covariance_matrix)

number_samples = 500
samples = x._get_sample(number_samples)
for sample in samples[x].data:
    plt.scatter(sample[0, 0, 0], sample[0, 1, 0], c="b")
plt.show()