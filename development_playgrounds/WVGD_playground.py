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
x = NormalVariable(theta**2, scale=0.2, name="x")
model = ProbabilisticModel([x, theta])

# Generate data
N = 3
theta_real = 0.1
x_real = NormalVariable(theta_real**2, 0.2, "x")
data = x_real._get_sample(number_samples=N)

# Observe data
x.observe(data[x_real][:, 0, :])

# Variational model
num_particles = 4
initial_locations = [np.random.normal(0., 0.1)
                     for _ in range(num_particles)]

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
                        number_post_samples=8000000)
inference.perform_inference(model,
                            inference_method=inference_method,
                            number_iterations=1000,
                            number_samples=50,
                            optimizer="SGD",
                            lr=0.0001,
                            posterior_model=particles,
                            pretraining_iterations=0)
loss_list = model.diagnostics["loss curve"]

# Local variational models
plt.plot(loss_list)
plt.show()

# Samples
print(inference_method.weights)
M = 8000
samples = [reformat_sample_to_pandas(sampler._get_sample(M, max_itr=np.inf)) for sampler in inference_method.sampler_model]
ensemble_histogram(samples,
                   variable="theta",
                   weights=inference_method.weights,
                   bins=50)
plt.show()

#print([p.get_sample(1) for p in particles])
#print(initial_locations)

#samples = Qtheta.get_sample(50)
#print(samples)