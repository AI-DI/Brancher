import numpy as np

import chainer
import chainer.links as L
import chainer.functions as F

from brancher.variables import RootVariable, RandomVariable, ProbabilisticModel
from brancher.standard_variables import NormalVariable, EmpiricalVariable, RandomIndices
from brancher.functions import BrancherFunction
import brancher.functions as BF

## Data ##
dataset_size = 100
number_dimensions = 1
dataset1 = np.random.normal(0, 1, (dataset_size, number_dimensions))

## Variables ##
indices = RandomIndices(dataset_size=dataset_size, batch_size=5, name="indices")
a = EmpiricalVariable(dataset1, indices=indices, name='a', is_observed=True)
b = EmpiricalVariable(dataset1, indices=indices, name='a', is_observed=True)

model = ProbabilisticModel([a, b])


## Sample ##
samples = model._get_sample(2)

print(samples[a])
print(samples[b])