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

import sys
sys.path
sys.path.append('/home/luca/GitRepositories/gan-pretrained-pytorch/cifar10_dcgan')

num_gpu = 1 if torch.cuda.is_available() else 0

# load the models
from dcgan import Discriminator, Generator

D = Discriminator(ngpu=1).eval()
G = Generator(ngpu=1).eval()

# load weights
D.load_state_dict(torch.load('/home/luca/GitRepositories/gan-pretrained-pytorch/cifar10_dcgan/weights/netD_epoch_199.pth', map_location=torch.device('cpu')))
G.load_state_dict(torch.load('/home/luca/GitRepositories/gan-pretrained-pytorch/cifar10_dcgan/weights/netG_epoch_199.pth', map_location=torch.device('cpu')))
if torch.cuda.is_available():
    D = D.cuda()
    G = G.cuda()

batch_size = 25
latent_size = 100

from brancher.variables import ProbabilisticModel
from brancher.standard_variables import NormalVariable, DeterministicVariable
import brancher.functions as BF

n_itr = 2000

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
    G = G.cpu()
    decoder = BF.BrancherFunction(lambda x: G(x))
    h_size = 100
    W1 = DeterministicVariable(np.random.normal(0., 0.5, (h_size, h_size)), "W1", learnable=False)
    W2 = DeterministicVariable(np.random.normal(0., 0.5, (h_size, h_size)), "W2", learnable=False)
    #V = DeterministicVariable(np.random.normal(0., 1., (100, h_size)), "V", learnable=False)

    f = lambda z, W: z + BF.tanh(BF.matmul(W, z))
    F = lambda z, W1, W2: f(f(z, W1), W2)

    measurement_noise = 0.5 #1.5
    z = [NormalVariable(np.zeros((h_size, 1)),
                        np.ones((h_size, 1)),
                        "z0", learnable=False)]
    img = [DeterministicVariable(decoder(BF.reshape(z[0], (100, 1, 1))), "img0", learnable=False)]
    x = [NormalVariable(img[0],
                        measurement_noise*np.ones((3, 32, 32)),
                        "x0", learnable=False)]

    T = 10
    t_cond = lambda t: t < 3 or t > T - 3
    driving_noise = 0.05
    for t in range(1, T):
      z.append(NormalVariable(F(z[-1], W1, W2),
                              driving_noise*np.ones((h_size, 1)),
                              "z{}".format(t), learnable=False))
      img.append(DeterministicVariable(decoder(BF.reshape(z[-1], (100, 1, 1))), "img{}".format(t), learnable=False))
      if t_cond(t):
        x.append(NormalVariable(img[-1],
                                measurement_noise*np.ones((3, 32, 32)),
                                "x{}".format(t), learnable=False))
    model = ProbabilisticModel(x + z + img)

    samples = model._get_sample(1)

    imagesGT.append([np.reshape(samples[imgt].detach().numpy(), (3, 32, 32))
                     for imgt in img])
    imagesNoise.append([np.reshape(samples[xt].detach().numpy(), (3, 32, 32))
                        for xt in x])

    # Observe model
    [xt.observe(samples[xt].detach().numpy()[0, :, :, :, :]) for xt in x]

    #### 1 ASDI ####

    PCoperator = lambda x, alpha, l: BF.sigmoid(l)*x + (1 - BF.sigmoid(l))*alpha

    Qz = [NormalVariable(np.zeros((h_size, 1)),
                         np.ones((h_size, 1)),
                         "z0", learnable=True)]
    Qalpha = []
    Qlambda = []
    for t in range(1, T):
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
                                lr=0.05)

    loss_list1.append(model.diagnostics["loss curve"])

    im_list1 = []
    for _ in range(5):
        psamples = model._get_posterior_sample(1)
        im_list1.append([np.reshape(psamples[img[t]].detach().numpy(), (3, 32, 32))
                         for t in range(T)])
    images1.append(im_list1)
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
                                lr=0.05)

    loss_list2.append(model.diagnostics["loss curve"])

    im_list2 = []
    for _ in range(5):
        psamples = model._get_posterior_sample(1)
        im_list2.append([np.reshape(psamples[img[t]].detach().numpy(), (3, 32, 32))
                         for t in range(T)])
    images2.append(im_list2)


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

    im_list3 = []
    for _ in range(5):
        psamples = model._get_posterior_sample(1)
        im_list3.append([np.reshape(psamples[img[t]].detach().numpy(), (3, 32, 32))
                         for t in range(T)])
    images3.append(im_list3)

d = {"Loss": [loss_list1, loss_list2, loss_list3], "Images": [images1, images2, images3], "Truth images": {"GT": imagesGT, "Noisy": imagesNoise}}

import pickle
with open('CIFAR_bridge_results.pickle', 'wb') as f:
    pickle.dump(d, f)

for rep in range(N_rep):
    plt.plot(loss_list1[rep])
    plt.plot(loss_list2[rep])
    plt.plot(loss_list3[rep])
    plt.show()

R, C = 1, T
for i, xi in enumerate(images1[0]):
  xi = xi.transpose((1, 2, 0))
  plt.subplot(R, C, i + 1)
  plt.imshow(xi, interpolation='bilinear')
plt.show()

