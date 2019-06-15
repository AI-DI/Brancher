import matplotlib.pyplot as plt
import numpy as np

from brancher.variables import ProbabilisticModel
from brancher.standard_variables import BetaVariable, BinomialVariable
from brancher import inference

#Real model
number_samples = 1
p_real = 0.8
k_real = BinomialVariable(number_samples, p=p_real, name="k")

# betaNormal/Binomial model
p = BetaVariable(1., 1., "p")
k = BinomialVariable(number_samples, p=p, name="k")
model = ProbabilisticModel([k])

# Generate data
data = k_real._get_sample(number_samples=50)

# Observe data
k.observe(data[k_real][:, 0, :])

# Variational distribution
Qp = BetaVariable(1., 1., "p", learnable=True)
model.set_posterior_model(ProbabilisticModel([Qp]))

# Inference
inference.perform_inference(model,
                            number_iterations=3000,
                            number_samples=100,
                            lr=0.01,
                            optimizer='Adam')
loss_list = model.diagnostics["loss curve"]

# Statistics
p_posterior_samples = model._get_posterior_sample(2000)[p].cpu().detach().numpy().flatten()

# Two subplots, unpack the axes array immediately
f, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(np.array(loss_list))
ax1.set_title("Convergence")
ax1.set_xlabel("Iteration")
ax2.hist(p_posterior_samples, 25)
ax2.axvline(x=p_real, lw=2, c="r")
ax2.set_title("Posterior samples (b)")
ax2.set_xlim(0,1)
plt.show()

