import numpy as np
import matplotlib.pyplot as plt

from brancher.variables import RootVariable, ProbabilisticModel
from brancher.standard_variables import NormalVariable, DeterministicVariable, CategoricalVariable, EmpiricalVariable, RandomIndices
from brancher import inference
import brancher.functions as BF

from brancher.particle_inference_tools import VoronoiSet
from brancher import inference
from brancher.inference import WassersteinVariationalGradientDescent as WVGD

#import brancher.config as cfg
#cfg.set_device('gpu')
#print(cfg.device)

# Data
import torchvision

ELBO_list = []
num_iterations = 25
for iteration in range(num_iterations):
    # Data
    image_size = 28
    num_classes = 10
    train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=None)
    test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=None)
    dataset_size = 500  #
    indices = np.random.choice(range(len(train)), dataset_size)  # le
    test_indices = np.random.choice(range(len(test)), dataset_size)# n(train)
    particles_ELBO = []
    for num_particles in [3, 1]:
        input_variable = np.reshape(train.train_data.numpy()[indices, :], newshape=(dataset_size, 1, image_size, image_size))
        output_labels = train.train_labels.numpy()[indices]

        # Test set data
        input_variable_test = np.reshape(test.test_data.numpy()[test_indices, :],
                                    newshape=(dataset_size, 1, image_size, image_size))
        output_labels_test = test.test_labels.numpy()[test_indices]

        # Data sampling model
        minibatch_size = 50
        minibatch_indices = RandomIndices(dataset_size=dataset_size, batch_size=minibatch_size,
                                          name="indices", is_observed=True)
        x = EmpiricalVariable(input_variable, indices=minibatch_indices,
                              name="x", is_observed=True)
        labels = EmpiricalVariable(output_labels, indices=minibatch_indices,
                                   name="labels", is_observed=True)

        # Test data
        x_test = EmpiricalVariable(input_variable_test, indices=minibatch_indices,
                              name="x", is_observed=True)
        labels_test = EmpiricalVariable(output_labels_test, indices=minibatch_indices,
                                        name="labels", is_observed=True)

        # Forward pass
        in_channels = 1
        out_channels = 10
        image_size = 28
        Wk = NormalVariable(loc=np.zeros((out_channels, in_channels, 2, 2)),
                            scale=np.ones((out_channels, in_channels, 2, 2)),
                            name="Wk")
        z = DeterministicVariable(BF.mean(BF.relu(BF.conv2d(x, Wk, stride=1)), (2, 3)), name="z")
        Wl = NormalVariable(loc=np.zeros((num_classes, out_channels)),
                            scale=np.ones((num_classes, out_channels)),
                            name="Wl")
        b = NormalVariable(loc=np.zeros((num_classes, 1)),
                           scale=np.ones((num_classes, 1)),
                           name="b")
        reshaped_z = BF.reshape(z, shape=(out_channels, 1))
        k = CategoricalVariable(logits=BF.linear(reshaped_z, Wl, b),
                        name="k")

        # Probabilistic model
        model = ProbabilisticModel([k])

        # Observations
        k.observe(labels)

        # Variational model
        #num_particles = 2 #10
        wk_locations = [np.random.normal(0., 0.1, (out_channels, in_channels, 2, 2)) for _ in range(num_particles)]
        wl_locations = [np.random.normal(0., 0.1, (num_classes, out_channels)) for _ in range(num_particles)]
        b_locations = [np.random.normal(0., 0.1, (num_classes, 1)) for _ in range(num_particles)]
        particles = [ProbabilisticModel([RootVariable(wk, name="Wk", learnable=True),
                                         RootVariable(wl, name="Wl", learnable=True),
                                         RootVariable(b, name="b", learnable=True)])
                     for wk, wl, b in zip(wk_locations, wl_locations, b_locations)]

        # Importance sampling distributions
        variational_samplers = [ProbabilisticModel([NormalVariable(wk, 0.1 + 0*wk, name="Wk", learnable=True),
                                                    NormalVariable(wl, 0.1 + 0*wl, name="Wl", learnable=True),
                                                    NormalVariable(b, 0.1 + 0*b, name="b", learnable=True)])
                                for wk, wl, b in zip(wk_locations, wl_locations, b_locations)]

        # Inference
        inference_method = WVGD(variational_samplers=variational_samplers,
                                particles=particles,
                                number_post_samples=1000,
                                biased=False)
        inference.perform_inference(model,
                                    inference_method=inference_method,
                                    number_iterations=4000,
                                    number_samples=10,
                                    optimizer="Adam",
                                    lr=0.01,
                                    posterior_model=particles,
                                    pretraining_iterations=0)
        loss_list = model.diagnostics["loss curve"]
        plt.plot(loss_list)
        #plt.show()

        # ELBO
        ELBO = model.posterior_model.estimate_log_model_evidence(number_samples=1000)
        print("Iteration {}, #particles{}, ELBO{}".format(iteration,
                                                          num_particles,
                                                          ELBO))
        particles_ELBO.append(float(ELBO.detach().numpy()))
    ELBO_list.append(particles_ELBO)
        # Test ELBO
        #ELBO_test = model.posterior_model.estimate_log_model_evidence(number_samples=3000)
        #x.observe(x_test)
        #k.observe(labels_test)
        #print("Test data. Iteration {}, #particles{}, ELBO{}".format(iteration,
                                                          #num_particles,
                                                          #model.posterior_model.estimate_log_model_evidence(
                                                              #number_samples=3000)))

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