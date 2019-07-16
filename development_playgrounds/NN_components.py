import numpy as np
import matplotlib.pyplot as plt
import torchvision

# Data
image_size = 28
num_classes = 10
train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=None)
test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=None)
dataset_size = len(train)
input_variable = np.reshape(train.train_data.numpy(), newshape=(dataset_size, 1, image_size, image_size))
output_labels = train.train_labels.numpy()

from brancher.standard_variables import EmpiricalStandardVariable as Empirical
from brancher.standard_variables import RandomIndices

# Data sampling model
minibatch_size = 7

minibatch_indices = RandomIndices(dataset_size=dataset_size, batch_size=minibatch_size,
                                  name="indices", is_observed=True)

x = Empirical(input_variable, indices=minibatch_indices,
              name="x", is_observed=True)

labels = Empirical(output_labels, indices=minibatch_indices,
                   name="labels", is_observed=True)

from brancher import functions as BF

from brancher.standard_variables import NormalStandardVariable as Normal
from brancher.standard_variables import CategoricalStandardVariable as Categorical

in_channels = 1
out_channels = 5
image_size = 28
#x = Normal(loc=np.zeros((in_channels, image_size, image_size)),
#           scale=1.,
#           name="x")
Wk = Normal(loc=np.zeros((out_channels, in_channels, 3, 3)),
            scale=1.*np.ones((out_channels, in_channels, 3, 3)),
            name="Wk")
z = Normal(BF.conv2d(x, Wk, padding=1), 1., name="z")

num_samples = 6
z.get_sample(num_samples)["z"]

num_classes = 10
Wl = Normal(loc=np.zeros((num_classes, image_size*image_size*out_channels)),
            scale=1.*np.ones((num_classes, image_size*image_size*out_channels)),
            name="Wl")
b = Normal(loc=np.zeros((num_classes, 1)),
           scale=1.*np.ones((num_classes, 1)),
           name="b")
reshaped_z = BF.reshape(z, shape=(image_size*image_size*out_channels, 1))
k = Categorical(logits=BF.linear(reshaped_z, Wl, b),
                name="k")

k.observe(labels)

from brancher.inference import MAP
from brancher.inference import perform_inference
from brancher.variables import ProbabilisticModel

convolutional_model = ProbabilisticModel([k])

perform_inference(convolutional_model,
                  inference_method=MAP(),
                  number_iterations=1,
                  optimizer="Adam",
                  lr=0.0025)
loss_list = convolutional_model.diagnostics["loss curve"]
#plt.plot(loss_list)
#plt.show()

import torch

import torch


## PyTorch model ##
class PytorchNetwork(torch.nn.Module):
    def __init__(self):
        super(PytorchNetwork, self).__init__()
        out_channels = 5
        image_size = 28
        self.l1 = torch.nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=3, padding=1)
        self.f1 = torch.nn.ReLU()
        self.l2 = torch.nn.Linear(in_features=image_size ** 2 * out_channels, out_features=10)

    def __call__(self, x):
        h = self.f1(self.l1(x))
        h_shape = h.shape
        h = h.view((h_shape[0], np.prod(h_shape[1:])))
        logits = self.l2(h)
        return logits

network = PytorchNetwork()

brancher_network = BF.BrancherFunction(network)

# Data sampling model
minibatch_size = 4
minibatch_indices = RandomIndices(dataset_size=dataset_size, batch_size=minibatch_size,
                                  name="indices", is_observed=True)
x = Empirical(input_variable, indices=minibatch_indices,
              name="x", is_observed=True)
labels = Empirical(output_labels, indices=minibatch_indices,
                   name="labels", is_observed=True)

# Forward model #
k = Categorical(logits=brancher_network(x),
                name="k")
print(k.get_sample(1)["k"][0])
k.observe(labels)

from brancher.inference import MaximumLikelihood
from brancher.inference import perform_inference
from brancher.variables import ProbabilisticModel

convolutional_model = ProbabilisticModel([k])

perform_inference(convolutional_model,
                  inference_method=MaximumLikelihood(),
                  number_iterations=500,
                  optimizer="Adam",
                  lr=0.0025)
loss_list = convolutional_model.diagnostics["loss curve"]
plt.plot(loss_list)