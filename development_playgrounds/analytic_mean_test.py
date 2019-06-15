from brancher.variables import ProbabilisticModel
from brancher.standard_variables import NormalVariable, CauchyVariable, LaplaceVariable
import numpy as np
np.random.seed(0)

dim = 1
a = CauchyVariable(loc=np.random.normal(0, 1, (dim, dim )),
                   scale=2 + np.random.normal(0, 1, (dim, dim ))**2,
                   name="a", learnable=True)

b = CauchyVariable(loc=np.random.normal(0, 1, (dim, dim )),
                   scale=2 + np.random.normal(0, 1, (dim, dim ))**2,
                   name="b", learnable=True)

c = CauchyVariable(loc=a + b,
                   scale=1 + a**2,
                   name="c", learnable=True)

d = CauchyVariable(loc=c,
                   scale=1 + b**2,
                   name="d", learnable=True)

model = ProbabilisticModel([d])

#print(model.get_mean())
#print(model.get_variance())
#print(model.get_entropy())

n_samples = 5
ent_list = []
samp_ent_list = []
for itr in range(50):
    q_sample = model._get_sample(n_samples)
    entropy = sum([e.sum()/float(n_samples) for e in model._get_entropy(q_sample).values()])
    ent_list.append(float(entropy))
    sampled_entropy = -model.calculate_log_probability(q_sample).mean()
    samp_ent_list.append(float(sampled_entropy))

print("Semi-analytic: {} +- {}".format(np.mean(ent_list), np.std(ent_list)))
print("Stochastic: {} +- {}".format(np.mean(samp_ent_list), np.std(samp_ent_list)))
