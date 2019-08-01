import brancher.config as cfg
cfg.set_device('gpu')
print(cfg.device)

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
        minibatch_size = 30
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
                            scale=10*np.ones((out_channels, in_channels, 2, 2)),
                            name="Wk")
        z = DeterministicVariable(BF.mean(BF.relu(BF.conv2d(x, Wk, stride=1)), (2, 3)), name="z")
        Wl = NormalVariable(loc=np.zeros((num_classes, out_channels)),
                            scale=10*np.ones((num_classes, out_channels)),
                            name="Wl")
        b = NormalVariable(loc=np.zeros((num_classes, 1)),
                           scale=10*np.ones((num_classes, 1)),
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
                                number_post_samples=500,
                                biased=True)
        inference.perform_inference(model,
                                    inference_method=inference_method,
                                    number_iterations=1500, #4000
                                    number_samples=20, #10
                                    optimizer="Adam",
                                    lr=0.05,
                                    posterior_model=particles,
                                    pretraining_iterations=0)
        loss_list = model.diagnostics["loss curve"]
        plt.plot(loss_list)

        # ELBO
        ELBO = model.posterior_model.estimate_log_model_evidence(number_samples=500)
        print("Iteration {}, #particles{}, ELBO{}".format(iteration,
                                                          num_particles,
                                                          ELBO))
        try:
            particles_ELBO.append(float(ELBO.detach().numpy()))
        except TypeError:
            particles_ELBO.append(float(ELBO.detach().cpu().numpy()))
    ELBO_list.append(particles_ELBO)
