import matplotlib.pyplot as plt

from brancher.variables import ProbabilisticModel
from brancher.standard_variables import BernulliVariable, NormalVariable
import brancher.functions as BF
from brancher import inference
from brancher.inference import ReverseKL
from brancher.gradient_estimators import BlackBoxEstimator, Taylor1Estimator

#Model
z1 = BernulliVariable(logits=0., name="z1")
z2 = BernulliVariable(logits=0., name="z2")
y = NormalVariable(2*z1 + z2, 1., name="y")
model = ProbabilisticModel([y])

#Generate data
data = y.get_sample(20, input_values={z1: 1, z2: 0})
data.hist(bins=20)
plt.show()

#Observe data
y.observe(data)

#Variational Model
Qz1 = BernulliVariable(logits=0., name="z1", learnable=True)
Qz2 = BernulliVariable(logits=0., name="z2", learnable=True)
variational_model = ProbabilisticModel([Qz1, Qz2])
model.set_posterior_model(variational_model)

# Joint-contrastive inference
inference.perform_inference(model,
                            inference_method=ReverseKL(gradient_estimator=Taylor1Estimator),
                            number_iterations=600,
                            number_samples=20,
                            optimizer="SGD",
                            lr=0.001)
loss_list = model.diagnostics["loss curve"]

#Plot results
plt.plot(loss_list)
plt.show()

#Plot posterior
model.get_posterior_sample(200).hist(bins=20)
plt.show()