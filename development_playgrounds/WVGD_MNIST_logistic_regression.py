import chainer
import matplotlib.pyplot as plt
import numpy as np

from brancher.variables import RootVariable, ProbabilisticModel
from brancher.standard_variables import NormalVariable, TruncatedNormalVariable, CategoricalVariable, EmpiricalVariable, RandomIndices
from brancher import inference
import brancher.functions as BF

from brancher.particle_inference_tools import VoronoiSet
from brancher import inference
from brancher.inference import WassersteinVariationalGradientDescent as WVGD

# Data
number_pixels = 28*28
number_output_classes = 10
train, test = chainer.datasets.get_mnist()
#dataset_size = len(train)
dataset_size = 50
input_variable = np.array([np.reshape(image[0], newshape=(number_pixels, 1)) for image in train][0:dataset_size]).astype("float32")
output_labels = np.array([image[1]*np.ones((1, 1)) for image in train][0:dataset_size]).astype("int32")

# Data sampling model
minibatch_size = 50
minibatch_indices = RandomIndices(dataset_size=dataset_size, batch_size=minibatch_size, name="indices", is_observed=True)
x = EmpiricalVariable(input_variable, indices=minibatch_indices, name="x", is_observed=True)
labels = EmpiricalVariable(output_labels, indices=minibatch_indices, name="labels", is_observed=True)

# Architecture parameters
weights = NormalVariable(np.zeros((number_output_classes, number_pixels)),
                         10*np.ones((number_output_classes, number_pixels)), "weights")

# Forward pass
final_activations = BF.matmul(weights, x)
k = CategoricalVariable(softmax_p=final_activations, name="k")

# Probabilistic model
model = ProbabilisticModel([k])

# Observations
k.observe(labels)

# Variational model
number_particles = 2
initial_location_1 = np.random.normal(0., 1., (number_output_classes, number_pixels))
initial_location_2 = np.random.normal(0., 1., (number_output_classes, number_pixels))
particle_1 = RootVariable(initial_location_1, name="weights", learnable=True)
particle_2 = RootVariable(initial_location_2, name="weights", learnable=True)
particle_locations = [particle_1, particle_2]
particles = [ProbabilisticModel([l]) for l in particle_locations]

# Importance sampling distributions
voranoi_set = VoronoiSet(particle_locations) #TODO: Bug if you use variables instead of probabilistic models
variational_samplers = [ProbabilisticModel([TruncatedNormalVariable(mu=initial_location_1, sigma=0.1,
                                                truncation_rule=lambda a: voranoi_set(a, 0),
                                                name="weights", learnable=True)]),
                        ProbabilisticModel([TruncatedNormalVariable(mu=initial_location_2, sigma=0.1,
                                                truncation_rule=lambda a: voranoi_set(a, 1),
                                                name="weights", learnable=True)])]

# Inference
inference.perform_inference(model,
                            inference_method=WVGD(biased=True),
                            number_iterations=1000,
                            number_samples=50,
                            optimizer=chainer.optimizers.Adam(0.005),
                            posterior_model=particles,
                            sampler_model=variational_samplers,
                            pretraining_iterations=0)
loss_list = model.diagnostics["loss curve"]

# Local variational models
plt.plot(loss_list)
plt.show()

# Test accuracy
num_images = 2000
test_size = len(test)
test_indices = RandomIndices(dataset_size=test_size, batch_size=1, name="test_indices", is_observed=True)
test_images = EmpiricalVariable(np.array([np.reshape(image[0], newshape=(number_pixels, 1)) for image in test]).astype("float32"),
                                indices=test_indices, name="x_test", is_observed=True)
test_labels = EmpiricalVariable(np.array([image[1]*np.ones((1, 1))
                                          for image in test]).astype("int32"), indices=test_indices, name="labels", is_observed=True)
test_model = ProbabilisticModel([test_images, test_labels])

s = 0
model.set_posterior_model(variational_samplers[0])
scores_0 = []

test_image_list = []
test_label_list = []
for _ in range(num_images):
    test_sample = test_model._get_sample(1)
    test_image, test_label = test_sample[test_images], test_sample[test_labels]
    test_image_list.append(test_image)
    test_label_list.append(test_label)

for test_image, test_label in zip(test_image_list,test_label_list):
    model_output = np.reshape(np.mean(model._get_posterior_sample(10, input_values={x: test_image})[k].data, axis=0), newshape=(10,))
    output_label = int(np.argmax(model_output))
    scores_0.append(1 if output_label == int(test_label.data) else 0)
    s += 1 if output_label == int(test_label.data) else 0
print("Accuracy 0: {} %".format(100*s/float(num_images)))

s = 0
model.set_posterior_model(variational_samplers[1])
scores_1 = []
for test_image, test_label in zip(test_image_list,test_label_list):
    model_output = np.reshape(np.mean(model._get_posterior_sample(10, input_values={x: test_image})[k].data, axis=0), newshape=(10,))
    output_label = int(np.argmax(model_output))
    scores_1.append(1 if output_label == int(test_label.data) else 0)
    s += 1 if output_label == int(test_label.data) else 0
print("Accuracy 1: {} %".format(100*s/float(num_images)))

s = 0
scores_ne = []
for test_image, test_label in zip(test_image_list,test_label_list):

    model.set_posterior_model(variational_samplers[0])
    model_output0 = np.reshape(np.mean(model._get_posterior_sample(10, input_values={x: test_image})[k].data, axis=0), newshape=(10,))

    model.set_posterior_model(variational_samplers[1])
    model_output1 = np.reshape(np.mean(model._get_posterior_sample(10, input_values={x: test_image})[k].data, axis=0), newshape=(10,))

    model_output = 0.5*(model_output0 + model_output1)

    output_label = int(np.argmax(model_output))
    scores_ne.append(1 if output_label == int(test_label.data) else 0)
    s += 1 if output_label == int(test_label.data) else 0
print("Accuracy Naive Ensemble: {} %".format(100*s/float(num_images)))

corr = np.corrcoef(scores_0, scores_1)[0,1]
print("Correlation: {}".format(corr))

print("TO DO")