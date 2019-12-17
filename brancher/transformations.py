from abc import ABC, abstractmethod

import numpy as np
import torch


from brancher.standard_variables import DeterministicVariable
import brancher.functions as BF

from brancher.variables import var2link
from brancher.utilities import coerce_to_dtype


class Transformation(ABC):

    @abstractmethod
    def __call__(self, var):
        pass


class Exp(Transformation):

    def __call__(self, var):
        return DeterministicVariable(BF.exp(var), log_determinant=-var, name="Exp({})".format(var.name))


class Sigmoid(Transformation):

    def __call__(self, var):
        return DeterministicVariable(BF.sigmoid(var),
                                     log_determinant=-BF.log(BF.sigmoid(var)) - BF.log(1 - BF.sigmoid(var)),
                                     name="Sigmoid({})".format(var.name))


class Softmax(Transformation):
    pass #TODO: To be implemented


class Scaling(Transformation):

    def __init__(self, s, learnable=False):
        self.s = s
        self.learnable = learnable

    def __call__(self, var):
        return DeterministicVariable(self.s*var,
                                     log_determinant=-BF.log(BF.abs(DeterministicVariable(self.s,
                                                                                          learnable=self.learnable,
                                                                                          name="{}_scale".format(var.name)))),
                                     name="{}*{}".format(self.s, var.name))

class Bias(Transformation):

    def __init__(self, b, learnable=False):
        self.b = b
        self.learnable = learnable

    def __call__(self, var):
        return DeterministicVariable(var + self.b,
                                     name="{}+{}".format(var.name, self.b))


class TriangularLinear(Transformation):

    def __init__(self, v, mat_dim, shift=0.001, upper=False, learnable=False):
        self.v = v
        tri_matrix = BF.triangular_form(v)
        if upper:
            tri_matrix = BF.transpose(tri_matrix, 2, 1) #TODO: Dangerous, uses inconsistent indexing
        self.tri_matrix = tri_matrix
        self.diag_indices = np.diag_indices(mat_dim)
        self.shift = shift
        self.upper = upper
        self.learnable = learnable

    def __call__(self, var):
        output = BF.matmul(self.tri_matrix, var)
        log_det = -BF.sum(BF.log(BF.abs(self.tri_matrix[:, self.diag_indices[0], self.diag_indices[1]]) + self.shift), axis=1)
        return DeterministicVariable(output,
                                     log_determinant=log_det,
                                     name="L {}".format(var.name))


class PlanarFlow(Transformation):

    def __init__(self, w, u, b, learnable=False, shift=0.001):
        self.w = w
        self.u = u
        self.b = b
        self.shift = shift
        self.learnable = learnable

    def __call__(self, var):
        dot_output = BF.dot(self.w, var, reduce=False) + self.b
        output = var + self.u*BF.sigmoid(dot_output)
        d_sigmoid = lambda x: BF.sigmoid(x)*(1. - BF.sigmoid(x))
        psy = d_sigmoid(dot_output)*self.w
        log_det = -BF.log(BF.abs(1. + BF.dot(self.u, psy)) + self.shift)
        return DeterministicVariable(output,
                                     log_determinant=log_det,
                                     name="PlanarFlow {}".format(var.name))
