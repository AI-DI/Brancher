"""
Variables
---------
Module description
"""

from abc import ABC, abstractmethod
import operator

import numpy as np

from brancher.variables import Variable, PartialLink, ProbabilisticModel
from brancher.time_series_models import TimeSeriesModel, LatentTimeSeriesModel
from brancher.standard_variables import MultivariateNormalVariable, DeterministicVariable
from brancher.standard_variables import var2link
from brancher.standard_variables import Process
import brancher.functions as BF
from brancher.utilities import get_numerical_index_from_string

from brancher.pandas_interface import pandas_frame2timeseries_data

import torch


class StochasticProcess(Process):

    def __init__(self):
        self.has_observation_model = False
        self.observation_cond_dist = None
        self.active_submodel = None

    def __call__(self, query_points): #This creates a finite-dimensional instance of the process
        instance = self.get_joint_instance(query_points)
        if self.has_observation_model:
            instance = self._apply_observation_model(instance)
        if isinstance(instance, ProbabilisticModel) and self.has_posterior_instance:
            instance.set_posterior_model(self.active_submodel.posterior_model)
            if self.active_submodel.observed_submodel is not None:
                observed_variables = [DeterministicVariable(value=var._observed_value[:, 0, :],
                                                            name=var.name, is_observed=True)
                                      for var in self.active_submodel.observed_submodel.variables] #TODO: Work in progress with observed variables
                instance.posterior_model.add_variables(observed_variables)
        return instance

    def attach_observation_model(self, observation_cond_dist):
        self.observation_cond_dist = observation_cond_dist
        self.has_observation_model = True

    def _apply_observation_model(self, instance): #TODO work in progress
        assert self.has_observation_model, "The observation model has not been initialized"
        if isinstance(instance, Variable):
            observation_variable = self.observation_cond_dist(instance)
            model = self._construct_observed_model(observation_variable, instance)
        else:
            observation_variables = []
            for var in instance._input_variables:
                obs_var = self.observation_cond_dist(var)
                obs_var.name = obs_var.name + "_" + get_numerical_index_from_string(var.name)
                observation_variables.append(obs_var)
            model = self._construct_observed_model(observation_variables, instance)
        return model

    @abstractmethod
    def _construct_observed_model(self, observation_variables, instance):
        pass

    @abstractmethod
    def observe(self, data, query_points):
        pass

    def unobserve(self):
        if self.active_submodel is not None:
            [var.unobserve() for var in self.active_submodel]

    @property
    def has_posterior_instance(self):
        return self.active_submodel is not None and self.active_submodel.posterior_model is not None

    def _assert_posterior_instance(self):
        assert self.has_posterior_instance, "Posterior samples can only be obtained after performing inference"

    @abstractmethod
    def get_joint_instance(self, query_points):
        pass

    def _get_sample(self, number_samples, query_points, input_values={}):
        multivariate_variable = self(query_points)
        return multivariate_variable._get_sample(number_samples=number_samples,
                                                 input_values=input_values)

    def get_sample(self, number_samples, query_points, input_values={}):
        multivariate_variable = self(query_points)
        return multivariate_variable.get_sample(number_samples=number_samples,
                                                input_values=input_values)

    def _get_posterior_sample(self, number_samples, query_points, input_values={}):
        self._assert_posterior_instance()
        multivariate_variable = self(query_points)
        return multivariate_variable._get_posterior_sample(number_samples=number_samples,
                                                           input_values=input_values)

    def get_posterior_sample(self, number_samples, query_points, input_values={}):
        self._assert_posterior_instance()
        multivariate_variable = self(query_points)
        return multivariate_variable.get_posterior_sample(number_samples=number_samples,
                                                          input_values=input_values)

## Processes types ##
class ContinuousStochasticProcess(StochasticProcess):
    pass


class DiscreteStochasticProcess(StochasticProcess):
    pass


## Gaussian processes ##
class GaussianProcess(ContinuousStochasticProcess):

    def __init__(self, mean_function, covariance_function, name):
        self.mean_function = mean_function
        self.covariance_function = covariance_function
        self.name = name
        super().__init__()

    def get_joint_instance(self, query_points):
        x = var2link(query_points)
        return MultivariateNormalVariable(loc=self.mean_function(x),
                                          covariance_matrix=self.covariance_function(x),
                                          name=self.name + "(" + query_points.name + ")")

    def _construct_observed_model(self, observation_variables, instance):
        return ProbabilisticModel(observation_variables)


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


## Discrete timeseries processes ##

class DiscreteTimeSeries(DiscreteStochasticProcess):

    def __init__(self):
        super().__init__()

    def observe(self, data, query_points):
        assert len(data) == len(query_points), "The number of datapoints should be equal to the number of query points"
        data = pandas_frame2timeseries_data(data)
        self.active_submodel = self(query_points)

        if self.has_observation_model:
            variables_to_be_observed = self.active_submodel.observation_variables
        else:
            variables_to_be_observed = self.active_submodel.temporal_variables

        [var.observe(data_point) for var, data_point in zip(variables_to_be_observed, data)]

    def get_timeseries_sample(self, number_samples, query_points, input_values={}):
        multivariate_variable = self(query_points)
        return multivariate_variable.get_timeseries_sample(number_samples=number_samples,
                                                           input_values=input_values,
                                                           mode="prior")

    def get_posterior_timeseries_sample(self, number_samples, query_points, input_values={}):
        self._assert_posterior_instance()
        multivariate_variable = self(query_points)
        return multivariate_variable.get_timeseries_sample(number_samples=number_samples,
                                                           input_values=input_values,
                                                           mode="posterior")

    def _construct_observed_model(self, observation_variables, instance):
        return LatentTimeSeriesModel(temporal_variables=instance.temporal_variables,
                                     observation_variables=observation_variables,
                                     time_stamps=instance.time_stamps)


class MarkovProcess(DiscreteTimeSeries):

    def __init__(self, initial_values, cond_dist):
        self.number_past_time_steps = cond_dist.__code__.co_argcount
        if isinstance(initial_values, Variable):
            assert self.number_past_time_steps == 1, "The conditional distribution cond_dist should have a single argument since a single initial_value was given"
            initial_values = tuple([initial_values])
        assert self.number_past_time_steps == len(initial_values), "The initial value should be a tuple of variables with as many entries as the number of arguments in the conditional distribution"
        self.initial_value = initial_values
        self.cond_dist = cond_dist
        super().__init__()

    def get_joint_instance(self, query_points):
        assert isinstance(query_points, (int, range)), "The input query_points of a Markov process should be either an integer (time horizon) or a range"
        if isinstance(query_points, int):
            time_range = range(0, query_points)
        else:
            time_range = query_points
        variables = list(self.initial_value)
        for t in time_range:
            if t >  self.number_past_time_steps - 1:
                new_variable = self.cond_dist(*(variables[-self.number_past_time_steps:-1] + [variables[-1]]))
                new_variable.name = new_variable.name + "_" + str(t)
                variables.append(new_variable)
        return TimeSeriesModel(variables, time_range)

