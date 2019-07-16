from brancher.standard_variables import NormalStandardVariable, DeterministicStandardVariable
import brancher.functions as BF

from numpy import sin

##
a = DeterministicStandardVariable(1.5, 'a')
b = DeterministicStandardVariable(0.3, 'b')
c = DeterministicStandardVariable(0.3, 'c')
d = BF.sin(a + b**2)/(3*c)

##
print(d)