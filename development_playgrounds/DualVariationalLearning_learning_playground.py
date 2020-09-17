import matplotlib.pyplot as plt
import numpy as np

from brancher.standard_variables import NormalVariable, DeterministicVariable, DirichletVariable
from brancher.variables import ProbabilisticModel
import brancher.functions as BF
from brancher import inference
from brancher import geometric_ranges

import torch
import torch.nn as nn
import torch.nn.functional as F

# Inference network
class InferenceNet(nn.Module):

    def __init__(self, n_hidden=50):
        super(InferenceNet, self).__init__()
        self.fc1 = nn.Linear(7, n_hidden)  # 6*6 from image dimension
        self.fc2 = nn.Linear(n_hidden, 2)
        self.fc3 = nn.Linear(n_hidden, 2)

    def forward(self, r, w0, w1, mx, my, sx, sy):
        x = torch.cat((r, w0, w1, mx, my, sx, s), 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softplus(self.fc3(x))
        return x, y

net = InferenceNet()

# Variational bayesian update
brancher_net = BF.BrancherFunction(net)

def bayesian_update(r, w0, w1, mx, my, sx, sy):
    out = brancher_net(r, w0, w1, mx, my, sx, sy)
    return [out[0],
            out[1],
            out[2],
            out[3]]


# Model
T = 20

sigma = 0.1
mx = [DeterministicVariable(0., "mx_0")]
my = [DeterministicVariable(0., "my_0")]
sx = [DeterministicVariable(0., "sx_0")]
sy = [DeterministicVariable(0., "sy_0")]
mux = [NormalVariable(mx[0].value, sx[0].value, "mux_0")]
muy = [NormalVariable(my[0].value, sy[0].value, "muy_0")]
Qmux = [NormalVariable(mx[0], sx[0], "mux_0")]
Qmuy = [NormalVariable(my[0], sy[0], "muy_0")]
w = []
r = []
for t in range(T):

    # Reward
    w.append(DirichletVariable(np.ones((2, 1)), "w_{}".format(t), is_policy=True, learnable=True))
    r.append(NormalVariable((w[t][0]*mux[t] + w[t][1]*muy[t]), sigma*BF.sqrt(w[t][0]**2 + w[0][1]**2),
                            name="r_{}".format(t), is_reward=True, is_observed=True))

    # Prior update
    mux.append(NormalVariable(mx[t], sx[t], "mux_{}".format(t+1), is_observed=True))
    muy.append(NormalVariable(my[t],
                              sy[t],
                              "muy_{}".format(t+1), is_observed=True))

    # Variational Bayesian update
    new_mx, new_my, new_sx, new_sy = bayesian_update(r[t], w[t][0], w[t][1], mx, my, sx, sy)
    Qmux.append(NormalVariable(new_mx, new_sx, "mux_{}".format(t+1), is_observed=True))
    Qmuy.append(NormalVariable(new_my, new_sy, "muy_{}".format(t+1), is_observed=True))

    mx.append(new_mx)
    my.append(new_my)
    sx.append(new_sx)
    sy.append(new_sy)

model = ProbabilisticModel(w + r + mux + muy)
variational_filter = ProbabilisticModel(Qmux + Qmuy)

# Variational model

print(model.get_average_reward(10))

# Train control
num_itr = 3000
inference.perform_inference(model,
                            posterior_model=variational_filter,
                            number_iterations=num_itr,
                            number_samples=9,
                            optimizer="Adam",
                            lr=0.01)
reward_list = model.diagnostics["reward curve"] #TODO: Very important. Solve the trained determinant problem. (it should be possible to specify which parameter is trainable)

print(model.get_sample(20)[["r"]])

plt.plot(reward_list)
plt.show()
print(model.get_average_reward(15))