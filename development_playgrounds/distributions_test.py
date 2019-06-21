import numpy as np

from brancher.standard_variables import DirichletVariable, GeometricVariable, Chi2Variable, \
    GumbelVariable, HalfCauchyVariable, HalfNormalVariable, NegativeBinomialVariable, PoissonVariable, StudentTVariable, UniformVariable

a = DirichletVariable(concentration=np.ones((10, 10)), name="a")
print(a.get_sample(2))

b = Chi2Variable(3, "b")
print(b.get_sample(2))

c = GeometricVariable(logits=0, name="c")
print(c.get_sample(2))

d = GumbelVariable(loc=0, scale=1, name="d")
print(d.get_sample(2))

e = HalfCauchyVariable(scale=1, name="e")
print(e.get_sample(2))

f = HalfCauchyVariable(scale=1, name="f")
print(f.get_sample(2))

g = HalfNormalVariable(scale=1, name="g")
print(g.get_sample(2))

h = NegativeBinomialVariable(1, logits=0, name="h")
print(h.get_sample(2))

i = PoissonVariable(1, name="i")
print(i.get_sample(2))

j = StudentTVariable(1,0,1, name="j")
print(j.get_sample(2))

l = UniformVariable(1, 2, name="l")
print(l.get_sample(2))
