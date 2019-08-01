import numpy as np
import matplotlib.pyplot as plt
import chainer

from brancher.variables import RootVariable, ProbabilisticModel
from brancher.particle_inference_tools import VoronoiSet
from brancher.standard_variables import EmpiricalVariable, NormalVariable, LogNormalVariable
from brancher import inference
from brancher.inference import WassersteinVariationalGradientDescent as WVGD
from brancher.visualizations import ensemble_histogram
from brancher.pandas_interface import reformat_sample_to_pandas

# Model
dimensionality = 1
theta = NormalVariable(loc=0., scale=2., name="theta")
x = NormalVariable(theta ** 2, scale=0.2, name="x")
model = ProbabilisticModel([x, theta])

# Generate data
N = 3
theta_real = 0.1
x_real = NormalVariable(theta_real ** 2, 0.2, "x")
data = x_real._get_sample(number_samples=N)

# Observe data
x.observe(data[x_real][:, 0, :])

# Variational model
num_particles = 2
initial_locations = [-2, 2]

#initial_locations = [0, 0.1]
particles = [ProbabilisticModel([RootVariable(p, name="theta", learnable=True)])
             for p in initial_locations]

# Importance sampling distributions
variational_samplers = [ProbabilisticModel([NormalVariable(loc=location, scale=0.2,
                                                           name="theta", learnable=True)])
                        for location in initial_locations]

# Inference
inference_method = WVGD(variational_samplers=variational_samplers,
                        particles=particles,
                        biased=False,
                        number_post_samples=80000)
inference.perform_inference(model,
                            inference_method=inference_method,
                            number_iterations=1500,
                            number_samples=50,
                            optimizer="Adam",
                            lr=0.005,
                            posterior_model=particles,
                            pretraining_iterations=0)
loss_list = model.diagnostics["loss curve"]

# Local variational models
plt.plot(loss_list)
plt.show()

#Plot posterior
#from brancher.visualizations import plot_density
#plot_density(model.posterior_model, variables=["theta"])
#plt.show()

# ELBO
print(model.posterior_model.estimate_log_model_evidence(number_samples=10000))