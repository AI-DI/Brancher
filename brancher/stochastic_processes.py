"""
Variables
---------
Module description
"""

from abc import ABC, abstractmethod
import operator

import numpy as np

from brancher.variables import Variable, PartialLink
from brancher.standard_variables import MultivariateNormalVariable
from brancher.standard_variables import var2link
import brancher.functions as BF
from brancher.utilities import coerce_to_dtype

import torch


class StochasticProcess(ABC):

    @abstractmethod
    def __call__(self, query_points):
        pass


## Gaussian processes ##
class GaussianProcess(StochasticProcess):

    def __init__(self, mean_function, covariance_function, name):
        self.mean_function = mean_function
        self.covariance_function = covariance_function
        self.name = name

    def __call__(self, query_points):
        x = var2link(query_points)
        return MultivariateNormalVariable(loc=self.mean_function(x),
                                          covariance_matrix=self.covariance_function(x),
                                          name=self.name + "(" + query_points.name + ")")


class CovarianceFunction(ABC):

    def __init__(self, covariance):
        self.covariance = covariance

    def __call__(self, query_points1, query_points2=None):
        if not query_points2:
            query_points2 = query_points1
        query_grid = BF.batch_meshgrid(query_points1, query_points2)
        return self.covariance(query_grid[0], query_grid[1])

    def __add__(self, other):
        if isinstance(other, CovarianceFunction):
            return CovarianceFunction(covariance=lambda x, y: self.covariance(x, y) + other.covariance(x, y))
        elif isinstance(other, (int, float)):
            return CovarianceFunction(covariance=lambda x, y: self.covariance(x, y) + other)
        else:
            raise ValueError("Only covarianceFunctions and numbers can be summed to CovarianceFunctions")

    def __mul__(self, other):
        if isinstance(other, CovarianceFunction):
            return CovarianceFunction(covariance=lambda x, y: self.covariance(x, y)*other.covariance(x, y))
        if isinstance(other, (Variable, PartialLink)):
            other = var2link(other)
            return CovarianceFunction(covariance=lambda x, y: self.covariance(x, y) * other)
        elif isinstance(other, (int, float)):
            return CovarianceFunction(covariance=lambda x, y: self.covariance(x, y) * other)
        else:
            raise ValueError("Only covarianceFunctions and numbers can be multiplied with CovarianceFunctions")

    def __rmul__(self, other):
        return self.__mul__(other)


class SquaredExponentialCovariance(CovarianceFunction):

    def __init__(self, scale, jitter=0.):
        self.scale = var2link(scale)
        covariance = lambda x, y: BF.exp(-(x-y)**2/(2*scale**2)) + BF.delta(x, y)*jitter
        super().__init__(covariance=covariance)


class WhiteNoiseCovariance(CovarianceFunction):

    def __init__(self, magnitude, jitter=0.):
        self.magnitude = var2link(magnitude)
        covariance = lambda x, y: magnitude*BF.delta(x, y) + BF.delta(x, y)*jitter
        super().__init__(covariance=covariance)


class HarmonicCovariance(CovarianceFunction):

    def __init__(self, frequency, jitter=0.):
        self.frequency = var2link(frequency)
        covariance = lambda x, y: BF.cos(2*np.pi*self.frequency*(x - y)) + BF.delta(x, y)*jitter
        super().__init__(covariance=covariance)


class PeriodicCovariance(CovarianceFunction):

    def __init__(self, frequency, scale, jitter=0.):
        self.frequency = var2link(frequency)
        self.scale = var2link(scale)
        covariance = lambda x, y: BF.exp(-2*BF.sin(np.pi*self.frequency*(x - y))**2/scale**2) + BF.delta(x, y)*jitter
        super().__init__(covariance=covariance)


class ExponentialCovariance(CovarianceFunction):

    def __init__(self, scale, jitter=0.):
        self.scale = var2link(scale)
        covariance = lambda x, y: BF.exp(-BF.abs(x-y)/(scale)) + BF.delta(x, y)*jitter
        super().__init__(covariance=covariance)


class MeanFunction(ABC):

    def __init__(self, mean):
        self.mean = mean

    def __call__(self, query_points):
        return self.mean(query_points)

    def _apply_operator(self, other, op):
        """
        Args:

        Returns:
        """
        if isinstance(other, MeanFunction):
            return lambda x: op(self.mean(x), other.mean(x))
        elif isinstance(other, (int, float)):
            return lambda x: op(self.mean(x), other)

    def __add__(self, other):
        return self._apply_operator(other, operator.add)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._apply_operator(other, operator.sub)

    def __rsub__(self, other):
        return -1*self.__sub__(other)

    def __mul__(self, other):
        return self._apply_operator(other, operator.mul)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self._apply_operator(other, operator.truediv)

    def __rtruediv__(self, other):
        raise NotImplementedError

    def __pow__(self, other):
        return self._apply_operator(other, operator.pow)

    def __rpow__(self, other):
        raise NotImplementedError


class ConstantMean(MeanFunction):

    def __init__(self, value):
        value = value
        mean = lambda x: BF.delta(x, x)*value
        super().__init__(mean=mean)