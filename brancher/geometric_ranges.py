"""
---------
"""
from abc import ABC, abstractmethod
import numpy as np

import brancher.functions as BF
from brancher.utilities import partial_broadcast


class GeometricRange(ABC):

    @abstractmethod
    def forward_transform(self, x, dim):
        pass

    @abstractmethod
    def inverse_transform(self, x, dim):
        pass


class UnboundedRange(GeometricRange):

    def __init__(self):
        pass

    def forward_transform(self, x, dim):
        return x

    def inverse_transform(self, y, dim):
        return y


class Interval(GeometricRange):

    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def forward_transform(self, x, dim):
        return self.lower_bound + (self.upper_bound-self.lower_bound)*BF.sigmoid(x)

    def inverse_transform(self, y, dim):
        z = (y - self.lower_bound) / (self.upper_bound - self.lower_bound)
        return np.log(z / (1 - z))


class RightHalfLine(GeometricRange):

    def __init__(self, lower_bound):
        self.lower_bound = lower_bound

    def forward_transform(self, x, dim):
        return self.lower_bound + BF.softplus(x)

    def inverse_transform(self, y, dim):
        return np.log(np.exp(y - self.lower_bound) - 1)


class LeftHalfLine(GeometricRange):

    def __init__(self, upper_bound):
        self.lower_bound = upper_bound

    def forward_transform(self, x, dim):
        return self.lower_bound - BF.softplus(x)

    def inverse_transform(self, y, dim):
        return np.log(np.exp(-y + self.lower_bound) - 1)


class Simplex(GeometricRange):

    def forward_transform(self, x, dim):
        latent_p = BF.softplus(x)
        normalization = BF.sum(latent_p, axis=1, keepdims=True)
        normalization = BF.broadcast_to(normalization, latent_p.shape())
        return latent_p / normalization

    def inverse_transform(self, y, dim):
        return np.log(np.exp(y) - 1)


class PositiveDefiniteMatrix(GeometricRange): #TODO: Work in progress

    def forward_transform(self, x, dim):
        return BF.matmul(x, BF.transpose(x, -2, -1))

    def inverse_transform(self, y, dim):
        chol_factor = np.linalg.cholesky(y)
        return chol_factor


# class UpperTriangularPositiveDefiniteMatrix(GeometricRange): #TODO: Work in progress
#
#     def forward_transform(self, x, dim):
#         pass
#         #matrix_shape = x.shape[-2:]
#         #return BF.matmul(x, BF.transpose(x, -2,-1))
#
#     def inverse_transform(self, y, dim):
#         pass
#         #chol_factor = np.linalg.cholesky(y)
#         #return chol_factor

