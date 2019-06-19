import numpy as np

from brancher import functions as BF

from brancher.standard_variables import NormalVariable as Normal
from brancher.standard_variables import DeterministicVariable

in_channels = 4
out_channels = 5
a = Normal(loc=np.zeros((in_channels, 28, 28)),
           scale=1.,
           name="a")
W = Normal(loc=np.zeros((out_channels, in_channels, 3, 3)),
           scale=np.ones((out_channels, in_channels, 3, 3)),
           name="W")
y = Normal(BF.conv2d(a, W), 0.1, name="y")

samples = y.get_sample(9)["y"]
print(samples[0].shape)
print(len(samples))
