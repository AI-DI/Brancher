import numpy as np
import matplotlib.pyplot as plt

import torchvision

# Create the data
image_size  = 28
num_classes = 10

train = torchvision.datasets.MNIST(root='./data',   train=True,  download=True, transform=None)
test  = torchvision.datasets.MNIST(root='./dataSo', train=False, download=True, transform=None)

dataset_size   = len(train)
input_variable = np.reshape(train.train_data.numpy(), newshape=(dataset_size, image_size*image_size))
output_labels  = train.train_labels.numpy()

from brancher.variables import ProbabilisticModel

from brancher.standard_variables import EmpiricalVariable as Empirical
from brancher.standard_variables import DeterministicVariable as Deterministic
from brancher.standard_variables import NormalVariable as Normal
from brancher.standard_variables import CategoricalVariable as Categorical
from brancher.standard_variables import RandomIndices

import brancher.functions as BF

#Data sampling model
minibatch_size = 250

minibatch_indices = RandomIndices( dataset_size=dataset_size, batch_size=minibatch_size,
                                   name="indices", is_observed=True )

x = Empirical(input_variable, indices=minibatch_indices,
              name="x", is_observed=True )

labels = Empirical(output_labels, indices=minibatch_indices,
                   name="labels", is_observed=True)

# Neural network
input_size = 28*28
#hidden_size1 = 30
#hidden_size2 = 10
out_size = 10

# Weights
#W1 = Deterministic(np.random.normal(0., 0.1, (hidden_size1, input_size)), "W1", learnable=True)
#W2 = Deterministic(np.random.normal(0., 0.1, (hidden_size2, hidden_size1)), "W2", learnable=True)
#W3 = Deterministic(np.random.normal(0., 0.1, (out_size, hidden_size2)), "W3", learnable=True)
V = Deterministic(np.random.normal(0., 0.1, (out_size, input_size)), "W3", learnable=True)

#z1 = Deterministic(BF.relu(BF.matmul(W1, BF.reshape(x, shape=(input_size, 1)))), "z1")
#z2 = Deterministic(BF.relu(BF.matmul(W2, z1)), "z2")
#rho = Deterministic(0.1*BF.matmul(W3, z2), "rho")
rho = Deterministic(BF.matmul(V, x/255), "rho")
k = Categorical(logits=rho, name="k")

# Observe
k.observe(labels)
model = ProbabilisticModel([k])

# Train
from brancher.inference import MaximumLikelihood
from brancher.inference import perform_inference

perform_inference(model,
                  inference_method=MaximumLikelihood(),
                  number_iterations=150,
                  optimizer="Adam",
                  lr=0.01)
loss_list = model.diagnostics["loss curve"]
plt.plot(loss_list)
print(loss_list[-1])
plt.show()

# import torch
#
#
# class PytorchNetwork(torch.nn.Module):
#     def __init__(self):
#         super(PytorchNetwork, self).__init__()
#         out_channels = 5
#         image_size = 28
#         self.l1 = torch.nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=3, padding=1)
#         self.f1 = torch.nn.ReLU()
#         self.l2 = torch.nn.Linear(in_features=image_size ** 2 * out_channels, out_features=10)
#
#     def __call__(self, x):
#         h = self.f1(self.l1(x))
#         h_shape = h.shape
#         h = h.view((h_shape[0], np.prod(h_shape[1:])))
#         logits = self.l2(h)
#         return logits
#
#
# network = PytorchNetwork()
#
# ## Equivalent Brancher model ##
# brancher_network = BF.BrancherFunction(network)
#
# # Data sampling model #
# minibatch_size = 250
# minibatch_indices = RandomIndices(dataset_size=dataset_size, batch_size=minibatch_size,
#                                   name="indices", is_observed=True)
# x = Empirical( input_variable, indices=minibatch_indices,
#                name="x", is_observed=True)
# labels = Empirical(output_labels, indices=minibatch_indices,
#                    name="labels", is_observed=True)
#
# # Forward model #
# k = Categorical(logits=brancher_network(x),
#                 name="k")
# k.observe(labels)
#
# from brancher.inference import MaximumLikelihood
# from brancher.inference import perform_inference
# from brancher.variables import ProbabilisticModel
#
# model = ProbabilisticModel([k])
#
# perform_inference(model,
#                   inference_method=MaximumLikelihood(),
#                   number_iterations=100,
#                   optimizer="Adam",
#                   lr=0.001)
# loss_list = model.diagnostics["loss curve"]
# plt.plot(loss_list)
# print(loss_list[-1])
# plt.show()

N = 500

test_size = test.test_data.numpy().shape[0]
test_images = np.reshape(test.test_data.numpy(), newshape=(test_size, image_size*image_size))

pred_labels = np.argmax(model.get_sample(1, input_values={x: test_images[:N, :]})["k"][0], axis=1)
true_labels = test.test_labels.numpy()[:N]

s = 0
for p_l, l in zip(pred_labels, true_labels):
    if p_l == l:
        s += 1
print("Accuracy: {}".format(s/float(N)))
