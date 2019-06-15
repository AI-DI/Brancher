import matplotlib.pyplot as plt
import numpy as np

from brancher.variables import ProbabilisticModel
from brancher.standard_variables import BetaVariable, BinomialVariable
from brancher import inference
from brancher.visualizations import plot_posterior


# betaNormal/Binomial model
number_tosses = 1
p = BetaVariable(1., 1., "p")
k = BinomialVariable(number_tosses, probs=p, name="k")
model = ProbabilisticModel([k, p])

# Generate data
p_real = 0.8
data = model.get_sample(number_samples=30, input_values={p: p_real})

# Observe data
k.observe(data)

# Variational distribution
Qp = BetaVariable(1., 1., "p", learnable=True)
model.set_posterior_model(ProbabilisticModel([Qp]))

# Inference
inference.perform_inference(model,
                            number_iterations=1000,
                            number_samples=500,
                            lr=0.1,
                            optimizer='SGD')
loss_list = model.diagnostics["loss curve"]

#Plot loss
plt.plot(loss_list)
plt.title("Loss (negative ELBO)")
plt.show()

#Plot posterior
plot_posterior(model, variables=["p"])
plt.show()

