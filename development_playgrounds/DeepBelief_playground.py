import torch
import torch.nn as nn
import torchvision

import matplotlib.pyplot as plt
import numpy as np

from brancher.variables import RootVariable, ProbabilisticModel
from brancher.standard_variables import NormalVariable, EmpiricalVariable, BinomialVariable, DeterministicVariable, LogNormalVariable
from brancher import inference
from brancher.inference import ReverseKL
from brancher.gradient_estimators import Taylor1Estimator, PathwiseDerivativeEstimator, BlackBoxEstimator
import brancher.functions as BF

from brancher.config import device

# Data
image_size = 28*28
latent_size1 = 2
latent_size2 = 100

train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=None)
test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=None)
dataset_size = len(train)
#dataset = torch.Tensor(np.reshape(train.train_data.numpy(), newshape=(dataset_size, image_size, 1))).double().to(device)
dataset = np.reshape(train.train_data.numpy(), newshape=(dataset_size, image_size, 1))
data_mean = np.mean(dataset)
dataset = (dataset > data_mean).astype("int32")

## Encoder 1 ##
class EncoderArchitecture1(nn.Module):

    def __init__(self, image_size, latent_size2, hidden_size=100):
        super(EncoderArchitecture1, self).__init__()
        self.l1 = nn.Linear(image_size, hidden_size)
        self.f1 = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, latent_size2)  # Latent mean output
        self.l3 = nn.Linear(hidden_size, latent_size2)  # Latent log sd output
        self.softplus = nn.Softplus()

    def __call__(self, x):
        h0 = self.f1(self.l1(x.squeeze()))
        output_mean = self.l2(h0)
        output_log_sd = self.l3(h0)
        return {"mean": output_mean, "sd": self.softplus(output_log_sd) + 0.01}

## Encoder 2 ##
class EncoderArchitecture2(nn.Module):

    def __init__(self, latent_size1, latent_size2, hidden_size=50):
        super(EncoderArchitecture2, self).__init__()
        self.l1 = nn.Linear(latent_size2, hidden_size)
        self.f1 = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, latent_size1)  # Latent mean output
        self.l3 = nn.Linear(hidden_size, latent_size1)  # Latent log sd output
        self.softplus = nn.Softplus()

    def __call__(self, x):
        h0 = self.f1(self.l1(x.squeeze()))
        output_mean = self.l2(h0)
        output_log_sd = self.l3(h0)
        return {"mean": output_mean, "sd": self.softplus(output_log_sd) + 0.01}


## Decoder ##
class DecoderArchitecture1(nn.Module):

    def __init__(self, latent_size1, latent_size2, hidden_size=30):
        super(DecoderArchitecture1, self).__init__()
        self.l1 = nn.Linear(latent_size1, hidden_size)
        self.f1 = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, latent_size2) # Latent mean output
        #self.l3 = nn.Linear(hidden_size, latent_size2)  # Latent SD output
        #self.softplus = nn.Softplus()

    def __call__(self, x):
        h0 = self.f1(self.l1(x))
        output_mean = self.l2(h0)
        #output_log_sd = self.l3(h0)
        return {"mean": output_mean}

class DecoderArchitecture2(nn.Module):

    def __init__(self, latent_size2, image_size):
        super(DecoderArchitecture2, self).__init__()
        self.l1 = nn.Linear(latent_size2, image_size)

    def __call__(self, x):
        output_mean = self.l1(x)
        return {"mean": output_mean}


# Initialize encoder and decoders
encoder1 = BF.BrancherFunction(EncoderArchitecture1(image_size=image_size, latent_size2=latent_size2))
encoder2 = BF.BrancherFunction(EncoderArchitecture2(latent_size1=latent_size1, latent_size2=latent_size2))
decoder1 = BF.BrancherFunction(DecoderArchitecture1(latent_size1=latent_size1, latent_size2=latent_size2))
decoder2 = BF.BrancherFunction(DecoderArchitecture2(latent_size2=latent_size2, image_size=image_size))

# Generative model
z1sd = 1.
z2sd = 0.5 #0.01
z1 = NormalVariable(np.zeros((latent_size1,)), z1sd*np.ones((latent_size1,)), name="z1")
decoder_output1 = DeterministicVariable(decoder1(z1), name="decoder_output1")
z2 = NormalVariable(BF.relu(decoder_output1["mean"]), z2sd*np.ones((latent_size2,)), name="z2")
decoder_output2 = DeterministicVariable(decoder2(z2), name="decoder_output2")
x = BinomialVariable(total_count=1, logits=decoder_output2["mean"], name="x")
model = ProbabilisticModel([x, z1, z2])

# Amortized variational distribution
b_size = 10
Qx = EmpiricalVariable(dataset, batch_size=b_size, name="x", is_observed=True)
encoder_output1 = DeterministicVariable(encoder1(Qx), name="encoder_output1")
Qz2 = NormalVariable(encoder_output1["mean"], encoder_output1["sd"], name="z2")
encoder_output2 = DeterministicVariable(encoder2(encoder_output1["mean"]), name="encoder_output2")
Qz1 = NormalVariable(encoder_output2["mean"], encoder_output2["sd"], name="z1")
model.set_posterior_model(ProbabilisticModel([Qx, Qz1, Qz2]))

model.get_sample(1)
model.posterior_model.get_sample(1)

# Joint-contrastive inference
num_itr = 2000
inference.perform_inference(model,
                            inference_method=ReverseKL(gradient_estimator=PathwiseDerivativeEstimator),
                            number_iterations=num_itr,
                            number_samples=1,
                            optimizer="Adam",
                            lr=0.0005)
loss_list1 = model.diagnostics["loss curve"]

N_ELBO = 20
N_ELBO_ITR = 1
ELBO = 0
for n in range(N_ELBO_ITR):
    ELBO += (model.estimate_log_model_evidence(N_ELBO)/float(N_ELBO_ITR)).detach().numpy()
print(ELBO)


#
# sigmoid = lambda x: 1/(np.exp(-x) + 1)
# image_grid = []
# z_range = np.linspace(-3, 3, 30)
# for z1a in z_range:
#     image_row = []
#     for z1b in z_range:
#         sample = model.get_sample(1, input_values={z1: np.array([z1a, z1b])})
#         image = sigmoid(np.reshape(sample["decoder_output2"].values[0]["mean"], newshape=(28, 28)))
#         image_row += [image]
#     image_grid += [np.concatenate(image_row, axis=0)]
# image_grid = np.concatenate(image_grid, axis=1)
# plt.imshow(image_grid)
# plt.colorbar()
# plt.show()

# Initialize encoder and decoders
encoder1 = BF.BrancherFunction(EncoderArchitecture1(image_size=image_size, latent_size2=latent_size2))
encoder2 = BF.BrancherFunction(EncoderArchitecture2(latent_size1=latent_size1, latent_size2=latent_size2))
decoder1 = BF.BrancherFunction(DecoderArchitecture1(latent_size1=latent_size1, latent_size2=latent_size2))
decoder2 = BF.BrancherFunction(DecoderArchitecture2(latent_size2=latent_size2, image_size=image_size))

# Generative model
z1 = NormalVariable(np.zeros((latent_size1,)), z1sd*np.ones((latent_size1,)), name="z1")
decoder_output1 = DeterministicVariable(decoder1(z1), name="decoder_output1")
z2 = NormalVariable(BF.relu(decoder_output1["mean"]), z2sd*np.ones((latent_size2,)), name="z2")
decoder_output2 = DeterministicVariable(decoder2(z2), name="decoder_output2")
x = BinomialVariable(total_count=1, logits=decoder_output2["mean"], name="x")
model = ProbabilisticModel([x, z1, z2])

# Amortized variational distribution
Qx = EmpiricalVariable(dataset, batch_size=b_size, name="x", is_observed=True)
encoder_output1 = DeterministicVariable(encoder1(Qx), name="encoder_output1")
encoder_output2 = DeterministicVariable(encoder2(encoder_output1["mean"]), name="encoder_output2")

l0 = 0

Qlambda11 = RootVariable(l0*np.ones((latent_size1,)), 'lambda11', learnable=True)
Qlambda12 = RootVariable(l0*np.ones((latent_size1,)), 'lambda12', learnable=True)
Qz1 = NormalVariable((1 - BF.sigmoid(Qlambda11))*encoder_output2["mean"],
                     BF.sigmoid(Qlambda12) * z2sd + (1 - BF.sigmoid(Qlambda12)) * encoder_output2["sd"], name="z1")

Qdecoder_output1 = DeterministicVariable(decoder1(Qz1), name="Qdecoder_output1")

Qlambda21 = RootVariable(l0*np.ones((latent_size2,)), 'lambda2', learnable=True)
Qlambda22 = RootVariable(l0*np.ones((latent_size2,)), 'lambda3', learnable=True)
Qz2 = NormalVariable(BF.sigmoid(Qlambda21)*BF.relu(Qdecoder_output1["mean"]) + (1 - BF.sigmoid(Qlambda21))*encoder_output1["mean"],
                     BF.sigmoid(Qlambda22) * z2sd + (1 - BF.sigmoid(Qlambda22)) * encoder_output1["sd"], name="z2")


model.set_posterior_model(ProbabilisticModel([Qx, Qz1, Qz2]))

model.get_sample(1)
model.posterior_model.get_sample(1)

# Joint-contrastive inference
inference.perform_inference(model,
                            inference_method=ReverseKL(gradient_estimator=PathwiseDerivativeEstimator),
                            number_iterations=num_itr,
                            number_samples=1,
                            optimizer="Adam",
                            lr=0.0005)
loss_list2 = model.diagnostics["loss curve"]

ELBO = 0
for n in range(N_ELBO_ITR):
    ELBO += (model.estimate_log_model_evidence(N_ELBO)/float(N_ELBO_ITR)).detach().numpy()
print(ELBO)

#Plot results
plt.plot(loss_list1, label="MF")
plt.plot(loss_list2, label="Structured")
plt.legend(loc="best")
plt.show()

sigmoid = lambda x: 1/(np.exp(-x) + 1)
image_grid = []
z_range = np.linspace(-3, 3, 50)
for z1a in z_range:
    image_row = []
    for z1b in z_range:
        sample = model.get_sample(1, input_values={z1: np.array([z1a, z1b])})
        image = sigmoid(np.reshape(sample["decoder_output2"].values[0]["mean"], newshape=(28, 28)))
        image_row += [image]
    image_grid += [np.concatenate(image_row, axis=0)]
image_grid = np.concatenate(image_grid, axis=1)
plt.imshow(image_grid)
plt.colorbar()
plt.show()

pass