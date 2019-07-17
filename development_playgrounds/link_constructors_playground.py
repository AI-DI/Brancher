from brancher.standard_variables import NormalVariable, DeterministicVariable
import brancher.functions as BF

from numpy import sin

##
a = DeterministicVariable(1.5, 'a')
b = DeterministicVariable(0.3, 'b')
c = DeterministicVariable(0.3, 'c')
d = BF.sin(a + b**2)/(3*c)

##
print(d)