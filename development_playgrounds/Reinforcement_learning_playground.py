import matplotlib.pyplot as plt
import numpy as np

from brancher.standard_variables import NormalVariable, DeterministicVariable, DirichletVariable
from brancher.variables import ProbabilisticModel
import brancher.functions as BF
from brancher import inference
from brancher import geometric_ranges

x = DeterministicVariable(1., "x", is_observed=True)
y = NormalVariable(-1., 0.1, "y", is_observed=True)
z = NormalVariable(0., 0.1,  "z", is_observed=True)
w = DirichletVariable(np.ones((3, 1)), "w", is_policy=True, learnable=True)
r = DeterministicVariable((w[0]*x + w[1]*y + w[2]*z), "r", is_reward=True, is_observed=True)

model = ProbabilisticModel([w, x, y, z, r])

print(model.get_average_reward(10))

# Train control
num_itr = 3000
inference.perform_inference(model,
                            inference_method=inference.MaximumLikelihood(),
                            number_iterations=num_itr,
                            number_samples=9,
                            optimizer="Adam",
                            lr=0.01)
reward_list = model.diagnostics["reward curve"] #TODO: Very important. Solve the trained determinant problem. (it should be possible to specify which parameter is trainable)

print(model.get_sample(20)[["r"]])

plt.plot(reward_list)
plt.show()
print(model.get_average_reward(15))