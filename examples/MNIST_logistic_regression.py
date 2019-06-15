import matplotlib.pyplot as plt
import numpy as np

import torchvision

from brancher.variables import RootVariable, ProbabilisticModel
from brancher.standard_variables import NormalVariable, CategoricalVariable, EmpiricalVariable, RandomIndices
from brancher import inference
from brancher.inference import ReverseKL
from brancher.gradient_estimators import Taylor1Estimator, PathwiseDerivativeEstimator
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
minibatch_size = 15
minibatch_indices = RandomIndices(dataset_size=dataset_size, batch_size=minibatch_size, name="indices", is_observed=True)
x = EmpiricalVariable(input_variable, indices=minibatch_indices, name="x", is_observed=True)
labels = EmpiricalVariable(output_labels, indices=minibatch_indices, name="labels", is_observed=True)

# Architecture parameters
weights = NormalVariable(np.zeros((number_output_classes, number_pixels)),
                         10*np.ones((number_output_classes, number_pixels)), "weights")

# Forward pass
final_activations = BF.matmul(weights, x)
k = CategoricalVariable(logits=final_activations, name="k")

# Probabilistic model
model = ProbabilisticModel([k])

# Observations
k.observe(labels)

# Variational Model
Qweights = NormalVariable(np.zeros((number_output_classes, number_pixels)),
                          0.1*np.ones((number_output_classes, number_pixels)), "weights", learnable=True)
variational_model = ProbabilisticModel([Qweights])
model.set_posterior_model(variational_model)

# Inference
inference.perform_inference(model,
                            inference_method=ReverseKL(gradient_estimator=PathwiseDerivativeEstimator),
                            number_iterations=500,
                            number_samples=1,
                            optimizer="Adam",
                            lr=0.005)

# Loss Curve
plt.plot(model.diagnostics["loss curve"])
plt.show()

# Test accuracy
num_images = 2000
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
    model_output = model._get_posterior_sample(10, input_values={x: test_image})[k].cpu().detach().numpy()
    output_probs = np.reshape(np.mean(model_output, axis=0), newshape=(10,))
    s += 1 if int(np.argmax(model_output)) == int(test_label.cpu().detach().numpy()) else 0
print("Accuracy: {} %".format(100*s/float(num_images)))

#weight_map = variational_model._get_sample(1)[Qweights1].detach().numpy()[0, 0, 0, :]
#plt.imshow(np.reshape(weight_map, (28, 28)))
#plt.show()

