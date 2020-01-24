import os
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pylab
import numpy as np

# load the models
use_gpu = False
gan_model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub', 'DCGAN', pretrained=True, useGPU=use_gpu)

from brancher.variables import ProbabilisticModel
from brancher.standard_variables import NormalVariable, DeterministicVariable
import brancher.functions as BF

#import brancher.config as cfg
#cfg.set_device('gpu')
#print(cfg.device)

Gmodel = gan_model.netG.cpu()
decoder = BF.BrancherFunction(lambda x: Gmodel(x))

n_itr = 2000 #2000
image_size = 64
N_rep = 5
loss_list1 = []
loss_list2 = []
loss_list3 = []
imagesGT = []
imagesNoise = []
images1 = []
images2 = []
images3 = []
for rep in range(N_rep):
    h_size = 120
    W1 = DeterministicVariable(np.random.normal(0., 0.2, (h_size, h_size)), "W1", learnable=False)
    W2 = DeterministicVariable(np.random.normal(0., 0.2, (h_size, h_size)), "W2", learnable=False)
    #V = DeterministicVariable(np.random.normal(0., 1., (100, h_size)), "V", learnable=False)

    f = lambda z, W: z + BF.tanh(BF.matmul(W, z))
    F = lambda z, W1, W2: f(f(z, W1), W2)

    measurement_noise = 2. #1.5
    z = [NormalVariable(np.zeros((h_size, 1)),
                        np.ones((h_size, 1)),
                        "z0", learnable=False)]
    img = [DeterministicVariable(decoder(BF.reshape(z[0], (h_size, 1, 1))), "img0", learnable=False)]
    x = [NormalVariable(img[0],
                        measurement_noise*np.ones((3, image_size, image_size)),
                        "x0", learnable=False)]

    T = 10
    t_cond = lambda t: True
    driving_noise = 0.05
    for t in range(1, T):
      z.append(NormalVariable(F(z[-1], W1, W2),
                              driving_noise*np.ones((h_size, 1)),
                              "z{}".format(t), learnable=False))
      img.append(DeterministicVariable(decoder(BF.reshape(z[-1], (h_size, 1, 1))), "img{}".format(t), learnable=False))
      if t_cond(t):
        x.append(NormalVariable(img[-1],
                                measurement_noise*np.ones((3, image_size, image_size)),
                                "x{}".format(t), learnable=False))
    model = ProbabilisticModel(x + z + img)

    samples = model._get_sample(1)

    imagesGT.append([np.reshape(samples[img[t]].detach().numpy(), (3, image_size, image_size))
                   for t in range(T)])
    imagesNoise.append([np.reshape(samples[x[t]].detach().numpy(), (3, image_size, image_size))
                        for t in range(T)])

    # Observe model
    [xt.observe(samples[xt].detach().numpy()[0, :, :, :, :]) for xt in x]

    #### 1 ASDI ####

    PCoperator = lambda x, alpha, l: BF.sigmoid(l)*x + (1 - BF.sigmoid(l))*alpha

    Qz = [NormalVariable(np.zeros((h_size, 1)),
                         np.ones((h_size, 1)),
                         "z0", learnable=True)]
    Qalpha = []
    Qlambda = []
    for t in range(1,T):
      #Qlambda.append(DeterministicVariable(-1*np.ones((h_size, 1)), "lambda{}".format(t), learnable=True))
      Qlambda.append(DeterministicVariable(-1., "lambda{}".format(t), learnable=True))
      Qalpha.append(DeterministicVariable(np.zeros((h_size, 1)),"alpha{}".format(t), learnable=True))
      Qz.append(NormalVariable(PCoperator(F(Qz[-1], W1, W2), Qalpha[-1], Qlambda[-1]),
                               0.5*np.ones((h_size, 1)),
                               "z{}".format(t), learnable=True))
    variational_model = ProbabilisticModel(Qz)
    model.set_posterior_model(variational_model)

    # Inference

    from brancher import inference

    # Inference #
    inference.perform_inference(model,
                                number_iterations=n_itr,
                                number_samples=3,
                                optimizer="Adam",
                                lr=0.01)

    loss_list1.append(model.diagnostics["loss curve"])

    psamples = model._get_posterior_sample(1)

    images1.append([np.reshape(psamples[img[t]].detach().numpy(), (3, image_size, image_size))
                    for t in range(T)])
    # plt.show()

    #### 2 Mean field ####

    Qz = [NormalVariable(np.zeros((h_size, 1)),
                        np.ones((h_size, 1)),
                        "z0", learnable=True)]

    for t in range(1,T):
      Qz.append(NormalVariable(np.zeros((h_size, 1)),
                               np.ones((h_size, 1)),
                               "z{}".format(t), learnable=True))
    variational_model = ProbabilisticModel(Qz)
    model.set_posterior_model(variational_model)

    # Inference

    from brancher import inference

    # Inference #
    inference.perform_inference(model,
                                number_iterations=n_itr,
                                number_samples=3,
                                optimizer="Adam",
                                lr=0.01)

    loss_list2.append(model.diagnostics["loss curve"])

    psamples =model._get_posterior_sample(1)

    images2.append([np.reshape(psamples[img[t]].detach().numpy(), (3, image_size, image_size))
                    for t in range(T)])


    #3 Gaussian linear

    QV = DeterministicVariable(np.random.normal(0, 0., (h_size, h_size)), "V", learnable=True)

    Qz = [NormalVariable(np.zeros((h_size, 1)),
                        np.ones((h_size, 1)),
                        "z0", learnable=True)]

    for t in range(1,T):
      Qmu = DeterministicVariable(np.zeros((h_size, 1)), "mu{}".format(t), learnable=True)
      Qz.append(NormalVariable(BF.matmul(QV,Qz[-1]) + Qmu,
                               np.ones((h_size, 1)),
                               "z{}".format(t), learnable=True))
    variational_model = ProbabilisticModel(Qz)
    model.set_posterior_model(variational_model)

    # Inference

    from brancher import inference

    # Inference #
    inference.perform_inference(model,
                                number_iterations=n_itr,
                                number_samples=3,
                                optimizer="Adam",
                                lr=0.01)

    loss_list3.append(model.diagnostics["loss curve"])

    psamples =model._get_posterior_sample(1)

    images3.append([np.reshape(psamples[img[t]].detach().numpy(), (3, image_size, image_size))
                    for t in range(T)])

d = {"Loss": [loss_list1, loss_list2, loss_list3], "Images": [images1, images2, images3], "Truth images": {"GT": imagesGT, "Noisy": imagesNoise}}

import pickle
with open('F_filter_results.pickle', 'wb') as f:
    pickle.dump(d, f)

for rep in range(N_rep):
    plt.plot(loss_list1[rep])
    plt.plot(loss_list2[rep])
    plt.plot(loss_list3[rep])
    plt.show()
