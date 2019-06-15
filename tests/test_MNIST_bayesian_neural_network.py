import matplotlib.pyplot as plt
import numpy as np

import torchvision

from brancher.variables import RootVariable, ProbabilisticModel
from brancher.standard_variables import NormalVariable, CategoricalVariable, EmpiricalVariable, RandomIndices
from brancher import inference
import brancher.functions as BF

# Data
number_pixels = 28*28
number_output_classes = 10
train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=None)
test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=None)
dataset_size = len(train)
input_variable = np.reshape(train.train_data.numpy(), newshape=(dataset_size, number_pixels, 1))
output_labels = train.train_labels.numpy()

# Data sampling model
minibatch_size = 30
minibatch_indices = RandomIndices(dataset_size=dataset_size, batch_size=minibatch_size, name="indices", is_observed=True)
x = EmpiricalVariable(input_variable, indices=minibatch_indices, name="x", is_observed=True)
labels = EmpiricalVariable(output_labels, indices=minibatch_indices, name="labels", is_observed=True)

# Architecture parameters
number_hidden_units = 20
b1 = NormalVariable(np.zeros((number_hidden_units, 1)),
                    10*np.ones((number_hidden_units, 1)), "b1")
b2 = NormalVariable(np.zeros((number_output_classes, 1)),
                    10*np.ones((number_output_classes, 1)), "b2")
weights1 = NormalVariable(np.zeros((number_hidden_units, number_pixels)),
                          10*np.ones((number_hidden_units, number_pixels)), "weights1")
weights2 = NormalVariable(np.zeros((number_output_classes, number_hidden_units)),
                          10*np.ones((number_output_classes, number_hidden_units)), "weights2")

# Forward pass
hidden_units = BF.tanh(BF.matmul(weights1, x) + b1)
final_activations = BF.matmul(weights2, hidden_units) + b2
k = CategoricalVariable(softmax_p=final_activations, name="k")

# Probabilistic model
model = ProbabilisticModel([k])

# Observations
k.observe(labels)

# Variational Model
Qb1 = NormalVariable(np.zeros((number_hidden_units, 1)),
                     0.2*np.ones((number_hidden_units, 1)), "b1", learnable=True)
Qb2 = NormalVariable(np.zeros((number_output_classes, 1)),
                     0.2*np.ones((number_output_classes, 1)), "b2", learnable=True)
Qweights1 = NormalVariable(np.zeros((number_hidden_units, number_pixels)),
                           0.2*np.ones((number_hidden_units, number_pixels)), "weights1", learnable=True)
Qweights2 = NormalVariable(np.zeros((number_output_classes, number_hidden_units)),
                           0.2*np.ones((number_output_classes, number_hidden_units)), "weights2", learnable=True)
variational_model = ProbabilisticModel([Qb1, Qb2, Qweights1, Qweights2])
model.set_posterior_model(variational_model)

# Inference
inference.perform_inference(model,
                            number_iterations=2000,
                            number_samples=50,
                            optimizer='Adam', lr=0.005) #0.05

# Test accuracy
num_images = 500
test_size = len(test)
test_indices = RandomIndices(dataset_size=test_size, batch_size=1, name="test_indices", is_observed=True)
test_images = EmpiricalVariable(np.reshape(test.test_data.numpy(), newshape=(test.test_data.shape[0], number_pixels, 1)),
                           indices=test_indices, name="x_test", is_observed=True)
test_labels = EmpiricalVariable(test.test_labels.numpy(), indices=test_indices, name="labels", is_observed=True)
test_model = ProbabilisticModel([test_images, test_labels])

s = 0
for _ in range(num_images):
    test_sample = test_model._get_sample(1)
    test_image, test_label = test_sample[test_images], test_sample[test_labels]
    model_output = np.reshape(np.mean(model._get_posterior_sample(10, input_values={x: test_image})[k].cpu().detach().numpy(), axis=0), newshape=(10,))
    s += 1 if int(np.argmax(model_output)) == int(test_label.cpu().detach().numpy()) else 0
print("Accuracy: {} %".format(100*s/float(num_images)))

#weight_map = variational_model._get_sample(1)[Qweights1].data[0, 0, 0, :]
#plt.imshow(np.reshape(weight_map, (28, 28)))
#plt.show()

plt.plot(model.diagnostics["loss curve"])
plt.show()