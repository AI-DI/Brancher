import torch
import torch.nn as nn
import torchvision

import matplotlib.pyplot as plt
import numpy as np

from brancher.variables import RootVariable, ProbabilisticModel
from brancher.standard_variables import NormalVariable, EmpiricalVariable, BinomialVariable, DeterministicVariable, RandomIndices, CategoricalVariable
from brancher import inference
from brancher.inference import ReverseKL
from brancher.gradient_estimators import PathwiseDerivativeEstimator
import brancher.functions as BF

from brancher.config import device

# Data
image_size = 28*28
latent_size1 = 2
latent_size2 = 50
latent_size3 = 100

train = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=None)
test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=None)
dataset_size = len(train)
dataset = np.reshape(train.train_data.numpy(), newshape=(dataset_size, image_size, 1))
data_mean = np.mean(dataset)
dataset = (dataset > data_mean).astype("int32")
output_labels = train.train_labels.numpy()

num_classes = 10

## Encoder 1 ##
class EncoderArchitecture1(nn.Module):

    def __init__(self, image_size, latent_size3, hidden_size=120, noise_inpt_size=None):
        super(EncoderArchitecture1, self).__init__()
        self.l1 = nn.Linear(image_size, hidden_size)
        self.f1 = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, latent_size3)  # Latent mean output
        self.l3 = nn.Linear(hidden_size, latent_size3)  # Latent log sd output
        self.softplus = nn.Softplus()
        if noise_inpt_size:
            self.ln = nn.Linear(noise_inpt_size, hidden_size)
        else:
            self.ln = None

    def __call__(self, x, noise_in=None):
        h0 = self.l1(x.squeeze())
        if noise_in is not None:
            h0 += self.ln(noise_in)
        h0 = self.f1(h0)
        output_mean = self.l2(h0)
        output_log_sd = self.l3(h0)
        return {"mean": output_mean, "sd": self.softplus(output_log_sd) + 0.01}

## Encoder 2 ##
class EncoderArchitecture2(nn.Module):

    def __init__(self, latent_size2, latent_size3, hidden_size=70, noise_inpt_size=None):
        super(EncoderArchitecture2, self).__init__()
        self.l1 = nn.Linear(latent_size3, hidden_size)
        self.f1 = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, latent_size2)  # Latent mean output
        self.l3 = nn.Linear(hidden_size, latent_size2)  # Latent log sd output
        self.softplus = nn.Softplus()
        if noise_inpt_size:
            self.ln = nn.Linear(noise_inpt_size, hidden_size)
        else:
            self.ln = None

    def __call__(self, x, noise_in=None):
        h0 = self.l1(x.squeeze())
        if noise_in is not None:
            h0 += self.ln(noise_in)
        h0 = self.f1(h0)
        output_mean = self.l2(h0)
        output_log_sd = self.l3(h0)
        return {"mean": output_mean, "sd": self.softplus(output_log_sd) + 0.01}

## Encoder 3 ##
class EncoderArchitecture3(nn.Module):

    def __init__(self, latent_size1, latent_size2, hidden_size=70, noise_inpt_size=None):
        super(EncoderArchitecture3, self).__init__()
        self.l1 = nn.Linear(latent_size2, hidden_size)
        self.f1 = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, latent_size1)  # Latent mean output
        self.l3 = nn.Linear(hidden_size, latent_size1)  # Latent log sd output
        self.softplus = nn.Softplus()
        if noise_inpt_size:
            self.ln = nn.Linear(noise_inpt_size, hidden_size)
        else:
            self.ln = None

    def __call__(self, x, noise_in=None):
        h0 = self.l1(x.squeeze())
        if noise_in is not None:
            h0 += self.ln(noise_in)
        h0 = self.f1(h0)
        output_mean = self.l2(h0)
        output_log_sd = self.l3(h0)
        return {"mean": output_mean, "sd": self.softplus(output_log_sd) + 0.01}


## Decoder ##
class DecoderArchitecture1(nn.Module):

    def __init__(self, latent_size1, latent_size2, hidden_size=25):
        super(DecoderArchitecture1, self).__init__()
        self.l1 = nn.Linear(latent_size1, hidden_size)
        self.f1 = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, latent_size2) # Latent mean output

    def __call__(self, x):
        h0 = self.f1(self.l1(x))
        output_mean = self.l2(h0)
        #output_log_sd = self.l3(h0)
        return {"mean": output_mean}

class DecoderArchitecture2(nn.Module):

    def __init__(self, latent_size2, latent_size3, hidden_size=75):
        super(DecoderArchitecture2, self).__init__()
        self.l1 = nn.Linear(latent_size2, hidden_size)
        self.f1 = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, latent_size3) # Latent mean output

    def __call__(self, x):
        h0 = self.f1(self.l1(x))
        output_mean = self.l2(h0)
        #output_log_sd = self.l3(h0)
        return {"mean": output_mean}

class DecoderArchitecture3(nn.Module):

    def __init__(self, latent_size3, image_size):
        super(DecoderArchitecture3, self).__init__()
        self.l1 = nn.Linear(latent_size3, image_size)

    def __call__(self, x):
        output_mean = self.l1(x)
        return {"mean": output_mean}

class DecoderArchitectureLabel(nn.Module):

    def __init__(self, latent_size2, num_classes):
        super(DecoderArchitectureLabel, self).__init__()
        self.l1 = nn.Linear(latent_size2, num_classes)

    def __call__(self, x):
        output_logits = self.l1(x)
        return output_logits

N_repetitions = 5 #5
num_itr = 3000
N_ELBO = 10
N_ELBO_ITR = 20
b_size = 200

loss_list1 = []
loss_list2 = []
loss_list3 = []

for rep in range(N_repetitions):
    # # Initialize encoder and decoders
    encoder1 = BF.BrancherFunction(EncoderArchitecture1(image_size=image_size, latent_size3=latent_size3))
    encoder2 = BF.BrancherFunction(EncoderArchitecture2(latent_size2=latent_size2, latent_size3=latent_size3))
    encoder3 = BF.BrancherFunction(EncoderArchitecture3(latent_size1=latent_size1, latent_size2=latent_size2))

    decoder1 = BF.BrancherFunction(DecoderArchitecture1(latent_size1=latent_size1, latent_size2=latent_size2))
    decoder2 = BF.BrancherFunction(DecoderArchitecture2(latent_size2=latent_size2, latent_size3=latent_size3))
    decoder3 = BF.BrancherFunction(DecoderArchitecture3(latent_size3=latent_size3, image_size=image_size))
    decoderLabel = BF.BrancherFunction(DecoderArchitectureLabel(latent_size2=latent_size2, num_classes=num_classes))

    # # Generative model
    z1sd = 1.5  # 1
    z2sd = 0.25  # 0.25
    z3sd = 0.15
    z1 = NormalVariable(np.zeros((latent_size1,)), z1sd * np.ones((latent_size1,)), name="z1")
    decoder_output1 = DeterministicVariable(decoder1(z1), name="decoder_output1")
    z2 = NormalVariable(BF.relu(decoder_output1["mean"]), z2sd * np.ones((latent_size2,)), name="z2")
    label_logits = DeterministicVariable(decoderLabel(z2), "label_logits")
    labels = CategoricalVariable(logits=label_logits, name="labels")
    decoder_output2 = DeterministicVariable(decoder2(z2), name="decoder_output2")
    z3 = NormalVariable(BF.relu(decoder_output2["mean"]), z3sd * np.ones((latent_size3,)), name="z3")
    decoder_output3 = DeterministicVariable(decoder3(z3), name="decoder_output3")
    x = BinomialVariable(total_count=1, logits=decoder_output3["mean"], name="x")
    model = ProbabilisticModel([x, z1, z2, z3, labels])

    # Amortized variational distribution

    minibatch_indices = RandomIndices(dataset_size=dataset_size, batch_size=b_size,
                                      name="indices", is_observed=True)

    Qx = EmpiricalVariable(dataset, indices=minibatch_indices,
                           name="x", is_observed=True)

    Qlabels = EmpiricalVariable(output_labels, indices=minibatch_indices,
                                name="labels", is_observed=True)

    encoder_output1 = DeterministicVariable(encoder1(Qx), name="encoder_output1")
    Qz3 = NormalVariable(encoder_output1["mean"], encoder_output1["sd"], name="z3")
    encoder_output2 = DeterministicVariable(encoder2(encoder_output1["mean"]), name="encoder_output2")
    Qz2 = NormalVariable(encoder_output2["mean"], encoder_output2["sd"], name="z2")
    encoder_output3 = DeterministicVariable(encoder3(encoder_output2["mean"]), name="encoder_output3")
    Qz1 = NormalVariable(encoder_output3["mean"], encoder_output3["sd"], name="z1")
    model.set_posterior_model(ProbabilisticModel([Qx, Qz1, Qz2, Qz3, Qlabels]))

    # Joint-contrastive inference
    inference.perform_inference(model,
                                inference_method=ReverseKL(gradient_estimator=PathwiseDerivativeEstimator),
                                number_iterations=num_itr,
                                number_samples=1,
                                optimizer="Adam",
                                lr=0.0005)
    loss_list1.append(np.array(model.diagnostics["loss curve"]))

    ELBO1 = []
    for n in range(N_ELBO_ITR):
        ELBO1.append(model.estimate_log_model_evidence(N_ELBO).detach().numpy())
    print("MF ELBO: {} +- {}".format(np.mean(ELBO1), np.std(ELBO1) / np.sqrt(float(N_ELBO_ITR))))

    # ## Structered hierarchical model
    # # Initialize encoder and decoders
    # noise_inpt_size = 50
    # encoder1 = BF.BrancherFunction(
    #     EncoderArchitecture1(image_size=image_size, latent_size3=latent_size3, noise_inpt_size=noise_inpt_size))
    # encoder2 = BF.BrancherFunction(
    #     EncoderArchitecture2(latent_size2=latent_size2, latent_size3=latent_size3, noise_inpt_size=noise_inpt_size))
    # encoder3 = BF.BrancherFunction(
    #     EncoderArchitecture3(latent_size1=latent_size1, latent_size2=latent_size2, noise_inpt_size=noise_inpt_size))
    #
    # decoder1 = BF.BrancherFunction(DecoderArchitecture1(latent_size1=latent_size1, latent_size2=latent_size2))
    # decoder2 = BF.BrancherFunction(DecoderArchitecture2(latent_size2=latent_size2, latent_size3=latent_size3))
    # decoder3 = BF.BrancherFunction(DecoderArchitecture3(latent_size3=latent_size3, image_size=image_size))
    # decoderLabel = BF.BrancherFunction(DecoderArchitectureLabel(latent_size2=latent_size2, num_classes=num_classes))
    #
    # # Generative model
    # z1 = NormalVariable(np.zeros((latent_size1,)), z1sd * np.ones((latent_size1,)), name="z1")
    # decoder_output1 = DeterministicVariable(decoder1(z1), name="decoder_output1")
    # z2 = NormalVariable(BF.relu(decoder_output1["mean"]), z2sd * np.ones((latent_size2,)), name="z2")
    # label_logits = DeterministicVariable(decoderLabel(z2), "label_logits")
    # labels = CategoricalVariable(logits=label_logits, name="labels")
    # decoder_output2 = DeterministicVariable(decoder2(z2), name="decoder_output2")
    # z3 = NormalVariable(BF.relu(decoder_output2["mean"]), z3sd * np.ones((latent_size3,)), name="z3")
    # decoder_output3 = DeterministicVariable(decoder3(z3), name="decoder_output3")
    # x = BinomialVariable(total_count=1, logits=decoder_output3["mean"], name="x")
    # model = ProbabilisticModel([x, z1, z2, z3, labels])
    #
    # # Amortized variational distribution
    # minibatch_indices = RandomIndices(dataset_size=dataset_size, batch_size=b_size,
    #                                   name="indices", is_observed=True)
    #
    # Qx = EmpiricalVariable(dataset, indices=minibatch_indices,
    #                        name="x", is_observed=True)
    #
    # Qlabels = EmpiricalVariable(output_labels, indices=minibatch_indices,
    #                             name="labels", is_observed=True)
    #
    # noise_input = NormalVariable(np.zeros((noise_inpt_size,)), 0.02 * np.ones((noise_inpt_size,)), "noise_input",
    #                              learnable=True)
    # encoder_output1 = DeterministicVariable(encoder1(Qx, noise_input), name="encoder_output1")
    # Qz3 = NormalVariable(encoder_output1["mean"], encoder_output1["sd"], name="z3")
    # encoder_output2 = DeterministicVariable(encoder2(encoder_output1["mean"], noise_input), name="encoder_output2")
    # Qz2 = NormalVariable(encoder_output2["mean"], encoder_output2["sd"], name="z2")
    # encoder_output3 = DeterministicVariable(encoder3(encoder_output2["mean"], noise_input), name="encoder_output3")
    # Qz1 = NormalVariable(encoder_output3["mean"], encoder_output3["sd"], name="z1")
    # model.set_posterior_model(ProbabilisticModel([Qx, Qz1, Qz2, Qz3, Qlabels]))
    #
    # # Joint-contrastive inference
    # inference.perform_inference(model,
    #                             inference_method=ReverseKL(gradient_estimator=PathwiseDerivativeEstimator),
    #                             number_iterations=num_itr,
    #                             number_samples=1,
    #                             optimizer="Adam",
    #                             lr=0.0005)
    # loss_list3.append(np.array(model.diagnostics["loss curve"]))
    #
    # ELBO3 = []
    # for n in range(N_ELBO_ITR):
    #     ELBO3.append(model.estimate_log_model_evidence(N_ELBO).detach().numpy())
    # print("Hierarchical ELBO: {} +- {}".format(np.mean(ELBO3), np.std(ELBO3) / np.sqrt(float(N_ELBO_ITR))))
    #
    # Initialize encoder and decoders
    encoder1 = BF.BrancherFunction(EncoderArchitecture1(image_size=image_size, latent_size3=latent_size3))
    encoder2 = BF.BrancherFunction(EncoderArchitecture2(latent_size2=latent_size2, latent_size3=latent_size3))
    encoder3 = BF.BrancherFunction(EncoderArchitecture3(latent_size1=latent_size1, latent_size2=latent_size2))

    decoder1 = BF.BrancherFunction(DecoderArchitecture1(latent_size1=latent_size1, latent_size2=latent_size2))
    decoder2 = BF.BrancherFunction(DecoderArchitecture2(latent_size2=latent_size2, latent_size3=latent_size3))
    decoder3 = BF.BrancherFunction(DecoderArchitecture3(latent_size3=latent_size3, image_size=image_size))
    decoderLabel = BF.BrancherFunction(DecoderArchitectureLabel(latent_size2=latent_size2, num_classes=num_classes))

    # Generative model
    z1 = NormalVariable(np.zeros((latent_size1,)), z1sd * np.ones((latent_size1,)), name="z1")
    decoder_output1 = DeterministicVariable(decoder1(z1), name="decoder_output1")
    z2 = NormalVariable(BF.relu(decoder_output1["mean"]), z2sd * np.ones((latent_size2,)), name="z2")
    label_logits = DeterministicVariable(decoderLabel(z2), "label_logits")
    labels = CategoricalVariable(logits=label_logits, name="labels")
    decoder_output2 = DeterministicVariable(decoder2(z2), name="decoder_output2")
    z3 = NormalVariable(BF.relu(decoder_output2["mean"]), z3sd * np.ones((latent_size3,)), name="z3")
    decoder_output3 = DeterministicVariable(decoder3(z3), name="decoder_output3")
    x = BinomialVariable(total_count=1, logits=decoder_output3["mean"], name="x")
    model = ProbabilisticModel([x, z1, z2, z3, labels])

    #
    # Amortized variational distribution
    l0 = 0  # 1
    l1 = -1

    minibatch_indices = RandomIndices(dataset_size=dataset_size, batch_size=b_size,
                                      name="indices", is_observed=True)

    Qx = EmpiricalVariable(dataset, indices=minibatch_indices,
                           name="x", is_observed=True)

    Qlabels = EmpiricalVariable(output_labels, indices=minibatch_indices,
                                name="labels", is_observed=True)

    encoder_output1 = DeterministicVariable(encoder1(Qx), name="encoder_output1")
    encoder_output2 = DeterministicVariable(encoder2(encoder_output1["mean"]), name="encoder_output2")
    encoder_output3 = DeterministicVariable(encoder3(encoder_output2["mean"]), name="encoder_output3")

    Qlambda11 = RootVariable(l1 * np.ones((latent_size1,)), 'lambda11', learnable=True)
    Qlambda12 = RootVariable(l1 * np.ones((latent_size1,)), 'lambda12', learnable=True)
    Qz1 = NormalVariable((1 - BF.sigmoid(Qlambda11)) * encoder_output3["mean"],
                         BF.sigmoid(Qlambda12) * z2sd + (1 - BF.sigmoid(Qlambda12)) * encoder_output3["sd"], name="z1")

    Qdecoder_output1 = DeterministicVariable(decoder1(Qz1), name="Qdecoder_output1")

    Qlambda21 = RootVariable(l0 * np.ones((latent_size2,)), 'lambda21', learnable=True)
    Qlambda22 = RootVariable(l0 * np.ones((latent_size2,)), 'lambda22', learnable=True)
    Qz2 = NormalVariable(
        BF.sigmoid(Qlambda21) * BF.relu(Qdecoder_output1["mean"]) + (1 - BF.sigmoid(Qlambda21)) * encoder_output2[
            "mean"],
        BF.sigmoid(Qlambda22) * z2sd + (1 - BF.sigmoid(Qlambda22)) * encoder_output2["sd"], name="z2")

    Qdecoder_output2 = DeterministicVariable(decoder2(Qz2), name="Qdecoder_output2")

    Qlambda31 = RootVariable(l0 * np.ones((latent_size3,)), 'lambda31', learnable=True)
    Qlambda32 = RootVariable(l0 * np.ones((latent_size3,)), 'lambda32', learnable=True)
    Qz3 = NormalVariable(
        BF.sigmoid(Qlambda31) * BF.relu(Qdecoder_output2["mean"]) + (1 - BF.sigmoid(Qlambda31)) * encoder_output1[
            "mean"],
        BF.sigmoid(Qlambda32) * z3sd + (1 - BF.sigmoid(Qlambda32)) * encoder_output1["sd"], name="z3")

    model.set_posterior_model(ProbabilisticModel([Qx, Qz1, Qz2, Qz3, Qlabels]))

    model.get_sample(1)
    model.posterior_model.get_sample(1)

    # Joint-contrastive inference
    inference.perform_inference(model,
                                inference_method=ReverseKL(gradient_estimator=PathwiseDerivativeEstimator),
                                number_iterations=num_itr,
                                number_samples=1,
                                optimizer="Adam",
                                lr=0.002)
    loss_list2.append(np.array(model.diagnostics["loss curve"]))
    #
    ELBO2 = []
    for n in range(N_ELBO_ITR):
        ELBO2.append(model.estimate_log_model_evidence(N_ELBO).detach().numpy())
    print("PE ELBO: {} +- {}".format(np.mean(ELBO2), np.std(ELBO2) / np.sqrt(float(N_ELBO_ITR))))

loss_list1 = np.array([[float(l) for l in sublist] for sublist in loss_list1])
loss_list2 = np.array([[float(l) for l in sublist] for sublist in loss_list2])
loss_list3 = np.array([[float(l) for l in sublist] for sublist in loss_list3])

loss_mean1 = sum(loss_list1)/float(N_repetitions)
loss_mean2 = sum(loss_list2)/float(N_repetitions)
loss_mean3 = sum(loss_list3)/float(N_repetitions)

loss_se1 = np.std((loss_list1),axis=0)/np.sqrt(N_repetitions)
loss_se2 = np.std((loss_list2),axis=0)/np.sqrt(N_repetitions)
loss_se3 = np.std((loss_list3),axis=0)/np.sqrt(N_repetitions)

sigmoid = lambda x: 1/(np.exp(-x) + 1)
image_grid = []
label_grid = []
z_range = np.linspace(-4, 4, 20)
for z1a in z_range:
    image_row = []
    label_row = []
    for z1b in z_range:
        sample = model.get_sample(1, input_values={z1: np.array([z1a, z1b])})
        image = sigmoid(np.reshape(sample["decoder_output3"].values[0]["mean"], newshape=(28, 28)))
        image_row += [((z1a, z1b), image)]
        label_row += [np.argmax(np.reshape(sample["label_logits"].values[0], newshape=(num_classes,)))]
    image_grid += [image_row]
    label_grid += [label_row]

d = {"Loss": {"MF": loss_list1, "PC": loss_list2},
     "ELBO": {"MF": ELBO1, "PC": ELBO2},
     "Images": {"Images": image_grid, "Labels": label_grid}}

import pickle
with open('fMNISTnetworkLabels.pickle', 'wb') as f:
    pickle.dump(d, f)

#Plot results
# rng = range(2*num_itr)
# plt.plot(rng, loss_mean1, label="MF", c="r")
# plt.fill_between(rng, loss_mean1 - loss_se1, loss_mean1 + loss_se1, color="r", alpha=0.5)
# plt.plot(rng, loss_mean2, label="Structured (PC)", c="b")
# plt.fill_between(rng, loss_mean2 - loss_se2, loss_mean2 + loss_se2, color="b", alpha=0.5)
# plt.plot(rng, loss_mean3, label="Structured (Hierarchical)", c="g")
# plt.fill_between(rng, loss_mean3 - loss_se3, loss_mean3 + loss_se3, color="g", alpha=0.5)
# plt.legend(loc="best")
# plt.show()

