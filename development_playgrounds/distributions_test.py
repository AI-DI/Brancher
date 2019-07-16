import numpy as np

from brancher.standard_variables import DirichletStandardVariable, GeometricStandardVariable, Chi2StandardVariable, \
    GumbelStandardVariable, HalfCauchyStandardVariable, HalfNormalStandardVariable, NegativeBinomialStandardVariable, PoissonStandardVariable, StudentTStandardVariable, UniformStandardVariable, BernoulliStandardVariable

## Distributions and samples ##
a = DirichletStandardVariable(concentration=np.ones((10, 10)), name="a")
print(a.get_sample(2))

b = Chi2StandardVariable(3, "b")
print(b.get_sample(2))

c = GeometricStandardVariable(logits=0, name="c")
print(c.get_sample(2))

d = GumbelStandardVariable(loc=0, scale=1, name="d")
print(d.get_sample(2))

e = HalfCauchyStandardVariable(scale=1, name="e")
print(e.get_sample(2))

f = HalfCauchyStandardVariable(scale=1, name="f")
print(f.get_sample(2))

g = HalfNormalStandardVariable(scale=1, name="g")
print(g.get_sample(2))

h = NegativeBinomialStandardVariable(1, logits=0, name="h")
print(h.get_sample(2))

i = PoissonStandardVariable(1, name="i")
print(i.get_sample(2))

j = StudentTStandardVariable(1, 0, 1, name="j")
print(j.get_sample(2))

l = UniformStandardVariable(1, 2, name="l")
print(l.get_sample(2))

m = BernoulliStandardVariable(probs=0.5, name="m")
print(m.get_sample(2))

## Moments ##
print("Moments :", m.distribution.get_moments(0.1, 5, **{"probs": 0.1}))


## Taylor ##
#print("Taylor :", m.distribution.get_log_p_taylor_expansion(0, 5, **{"probs": 0.1}))

