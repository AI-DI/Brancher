import numpy as np
import matplotlib.pyplot as plt

from brancher.variables import RootVariable, ProbabilisticModel
from brancher.standard_variables import NormalVariable, DeterministicVariable, CategoricalVariable, EmpiricalVariable, RandomIndices
from brancher import inference
import brancher.functions as BF

from brancher.particle_inference_tools import VoronoiSet
from brancher import inference
from brancher.inference import WassersteinVariationalGradientDescent as WVGD

# Data
import torchvision

# Data
image_size = 28
num_classes = 10
train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=None)
test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=None)
dataset_size = 100 #len(train)
input_variable = np.reshape(train.train_data.numpy()[:dataset_size,:], newshape=(dataset_size, 1, image_size, image_size))
output_labels = train.train_labels.numpy()

# Data sampling model
minibatch_size = 7
minibatch_indices = RandomIndices(dataset_size=dataset_size, batch_size=minibatch_size,
                                  name="indices", is_observed=True)
x = EmpiricalVariable(input_variable, indices=minibatch_indices,
                      name="x", is_observed=True)
labels = EmpiricalVariable(output_labels, indices=minibatch_indices,
                           name="labels", is_observed=True)

# Forward pass
in_channels = 1
out_channels1 = 10
out_channels2 = 20
image_size = 28
Wk1 = NormalVariable(loc=np.zeros((out_channels1, in_channels, 3, 3)),
                     scale=np.ones((out_channels1, in_channels, 3, 3)),
                     name="Wk1")
Wk2 = NormalVariable(loc=np.zeros((out_channels2, out_channels1, 3, 3)),
                     scale=np.ones((out_channels2, out_channels1, 3, 3)),
                     name="Wk2")
z = DeterministicVariable(BF.mean(BF.conv2d(BF.relu(BF.conv2d(x, Wk1,
                                                              stride=2,
                                                              padding=0)), Wk2,
                                            stride=2,
                                            padding=0), (2, 3)), name="z")
Wl = NormalVariable(loc=np.zeros((num_classes, out_channels2)),
                    scale=np.ones((num_classes, out_channels2)),
                    name="Wl")
b = NormalVariable(loc=np.zeros((num_classes, 1)),
                   scale=np.ones((num_classes, 1)),
                   name="b")
reshaped_z = BF.reshape(z, shape=(out_channels2, 1))
k = CategoricalVariable(logits=BF.linear(reshaped_z, Wl, b),
                name="k")

# Probabilistic model
model = ProbabilisticModel([k])
samples = model.get_sample(10)

# Observations
k.observe(labels)

# Variational model
num_particles = 4 #10
wk1_locations = [np.random.normal(0., 1., (out_channels1, in_channels, 3, 3)) for _ in range(num_particles)]
wk2_locations = [np.random.normal(0., 1., (out_channels2, out_channels1, 3, 3)) for _ in range(num_particles)]
wl_locations = [np.random.normal(0., 1., (num_classes, out_channels2)) for _ in range(num_particles)]
b_locations = [np.random.normal(0., 1., (num_classes, 1)) for _ in range(num_particles)]
particles = [ProbabilisticModel([RootVariable(wk1, name="Wk1", learnable=True),
                                 RootVariable(wk2, name="Wk2", learnable=True),
                                 RootVariable(wl, name="Wl", learnable=True),
                                 RootVariable(b, name="b", learnable=True)])
             for wk1, wk2, wl, b in zip(wk1_locations, wk2_locations, wl_locations, b_locations)]

# Importance sampling distributions
variational_samplers = [ProbabilisticModel([NormalVariable(wk1, 1 + 0*wk1, name="Wk1", learnable=True),
                                            NormalVariable(wk2, 1 + 0*wk2, name="Wk2", learnable=True),
                                            NormalVariable(wl, 1 + 0*wl, name="Wl", learnable=True),
                                            NormalVariable(b, 1 + 0*b, name="b", learnable=True)])
                        for wk1, wk2, wl, b in zip(wk1_locations, wk2_locations, wl_locations, b_locations)]

# Inference
inference_method = WVGD(variational_samplers=variational_samplers,
                        particles=particles,
                        biased=False)
inference.perform_inference(model,
                            inference_method=inference_method,
                            number_iterations=2000,
                            number_samples=20,
                            optimizer="Adam",
                            lr=0.0025,
                            posterior_model=particles,
                            pretraining_iterations=0)
loss_list = model.diagnostics["loss curve"]
plt.plot(loss_list)
#plt.show()

# ELBO
print("#particles{}, ELBO{}".format(num_particles, model.posterior_model.estimate_log_model_evidence(number_samples=10000)))

# # Local variational models
# plt.plot(loss_list)
# plt.show()
#
# # Test accuracy
# num_images = 2000
# test_size = len(test)
# test_indices = RandomIndices(dataset_size=test_size, batch_size=1, name="test_indices", is_observed=True)
# test_images = EmpiricalVariable(np.array([np.reshape(image[0], newshape=(number_pixels, 1)) for image in test]).astype("float32"),
#                                 indices=test_indices, name="x_test", is_observed=True)
# test_labels = EmpiricalVariable(np.array([image[1] * np.ones((1, 1))
#                                           for image in test]).astype("int32"), indices=test_indices, name="labels", is_observed=True)
# test_model = ProbabilisticModel([test_images, test_labels])
#
# s = 0
# model.set_posterior_model(variational_samplers[0])
# scores_0 = []
#
# test_image_list = []
# test_label_list = []
# for _ in range(num_images):
#     test_sample = test_model._get_sample(1)
#     test_image, test_label = test_sample[test_images], test_sample[test_labels]
#     test_image_list.append(test_image)
#     test_label_list.append(test_label)
#
# for test_image, test_label in zip(test_image_list,test_label_list):
#     model_output = np.reshape(np.mean(model._get_posterior_sample(10, input_values={x: test_image})[k].data, axis=0), newshape=(10,))
#     output_label = int(np.argmax(model_output))
#     scores_0.append(1 if output_label == int(test_label.data) else 0)
#     s += 1 if output_label == int(test_label.data) else 0
# print("Accuracy 0: {} %".format(100*s/float(num_images)))
#
# s = 0
# model.set_posterior_model(variational_samplers[1])
# scores_1 = []
# for test_image, test_label in zip(test_image_list,test_label_list):
#     model_output = np.reshape(np.mean(model._get_posterior_sample(10, input_values={x: test_image})[k].data, axis=0), newshape=(10,))
#     output_label = int(np.argmax(model_output))
#     scores_1.append(1 if output_label == int(test_label.data) else 0)
#     s += 1 if output_label == int(test_label.data) else 0
# print("Accuracy 1: {} %".format(100*s/float(num_images)))
#
# s = 0
# scores_ne = []
# for test_image, test_label in zip(test_image_list,test_label_list):
#
#     model.set_posterior_model(variational_samplers[0])
#     model_output0 = np.reshape(np.mean(model._get_posterior_sample(10, input_values={x: test_image})[k].data, axis=0), newshape=(10,))
#
#     model.set_posterior_model(variational_samplers[1])
#     model_output1 = np.reshape(np.mean(model._get_posterior_sample(10, input_values={x: test_image})[k].data, axis=0), newshape=(10,))
#
#     model_output = 0.5*(model_output0 + model_output1)
#
#     output_label = int(np.argmax(model_output))
#     scores_ne.append(1 if output_label == int(test_label.data) else 0)
#     s += 1 if output_label == int(test_label.data) else 0
# print("Accuracy Naive Ensemble: {} %".format(100*s/float(num_images)))
#
# corr = np.corrcoef(scores_0, scores_1)[0,1]
# print("Correlation: {}".format(corr))
#
# print("TO DO")