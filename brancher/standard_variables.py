import numbers
from abc import ABC, abstractmethod

import numpy as np
import torch.nn as nn
import torch

import brancher.distributions as distributions
import brancher.functions as BF
import brancher.geometric_ranges as geometric_ranges
from brancher.variables import var2link, Variable, RootVariable, RandomVariable, PartialLink
from brancher.utilities import join_sets_list

from brancher.config import device


## Stochastic process base class ##
class Process(ABC):
    """
    Process is the superclass of stochastic processes
    """
    @abstractmethod
    def attach_observation_model(self, observation_cond_dist):
        pass


class Link(nn.ModuleList):
    """
    Links represent a dependency tree between variables. Links construct a pytorch module list from its arguments. When
    called it returns the values from each variable given the input values.

    Parameters
    ----------
    kwargs: Dictionary(str: brancher.Variables, numbers, numpy.ndarrays, torch.Tensors, or List/Tuple of
    brancher.Variables and brancher.PartialLinks). String variable pairs where the variables are converted to partial
    links.
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        modules = [link
                   for partial_link in kwargs.values()
                   for link in var2link(partial_link).links]
        super().__init__(modules) #TODO: asserts that specified links are valid pytorch modules

    def __call__(self, values):
        return {k: var2link(x).fn(values) for k, x in self.kwargs.items()}


class StandardVariable(RandomVariable):
    """
    Standard Variable superclass that constructs a variable or random process. The main purpose of this class is for the
    user to be able to easily construct variables that depend on other variables and implicit deterministic variables.

    Parameters
    ----------
    name: String. Name of this variable

    learnable: Bool. Set true if this variable is learnable by inference.

    ranges: Dictionary(str: brancher.GeometricRange). Dictionary of variable names and the ranges that apply on those
    variables.

    is_observed: Bool. Set true if this variable is observed.

    has_bias: Bool. Set true if bias variables should be constructed.

    **kwargs: Named variables list that define the input variables of this variable.
    """
    def __new__(cls, *args, **kwargs):
        """
        Method. Constructs a new StandardVariable or a Process if a Process is given as one of the arguments.

        Args:
            cls. Subclass of brancher.StandardVariable. The specific StandardVariable that needs to be initialized.

        Returns:
            brancher.StandardVariable or brancher.Process.
        """
        var = super(StandardVariable, cls).__new__(cls)
        var.__init__(*args, **kwargs)
        return var._construct_variable_or_process()


    def __init__(self, name, learnable, ranges, is_observed=False, has_bias=False, **kwargs):
        self._input = kwargs
        self.name = name
        self._evaluated = False
        self._observed = is_observed
        self._observed_value = None
        self._current_value = None
        if self._check_for_stochastic_process_arguments():
            return
        if has_bias:
            self.construct_biases(learnable, ranges, kwargs)
        self.construct_deterministic_parents(learnable, ranges, kwargs)
        self.parents = join_sets_list([var2link(x).vars for x in kwargs.values()])
        self.link = Link(**kwargs)
        self.ancestors = join_sets_list([self.parents] + [parent.ancestors for parent in self.parents])
        self.samples = None
        self.ranges = {}
        self.dataset = None
        self.has_random_dataset = False
        self.has_observed_value = False
        self.is_normalized = True
        self.silenced = False
        self.partial_links = {name: var2link(link) for name, link in kwargs.items()}

    def construct_deterministic_parents(self, learnable, ranges, kwargs):
        """
        Method. Constructs the deterministic variables for input variables that are numberic or numpy arrays. If a
        variable is an instance of brancher.Variable or brancher.PartialLink these should already have deterministic
        variables initialized.

        Args:
            learnable: Bool. Set true if the root variables should be learnable.

            ranges: Dictionary(str: brancher.GeometricRange). Dictionary of variable names and the ranges that apply on
            those variables.

            kwargs: Named variables list that define the input variables of this variable.

        Returns:
            None.
        """
        for parameter_name, value in kwargs.items():
            if not isinstance(value, (Variable, PartialLink)):
                if isinstance(value, np.ndarray):
                    dim = value.shape[0]
                elif isinstance(value, numbers.Number):
                    dim = 1
                else:
                    dim = []
                deterministic_parent = RootVariable(ranges[parameter_name].inverse_transform(value, dim),
                                                    self.name + "_" + parameter_name, learnable, is_observed=self._observed)
                kwargs.update({parameter_name: ranges[parameter_name].forward_transform(deterministic_parent, dim)})

    def construct_biases(self, learnable, ranges, kwargs):
        """
        Method. Constructs a bias variable for each variable in the input. Bias variables are deterministic variables
        that transform the input variables.

        Args:
            learnable: Bool. Set true if the biases should be learnable.

            ranges: Dictionary(str: brancher.GeometricRange). Dictionary of variable names and the ranges that apply on
            those variables.

            kwargs: Named variables list that define the input variables of this variable. For each variable in here a
            bias variable will be created.

        Returns:
            None.
        """
        for parameter_name, value in kwargs.items():
            if isinstance(value, (Variable, PartialLink, np.ndarray, numbers.Number)):
                if isinstance(value, np.ndarray):
                    dim = value.shape[0]
                elif isinstance(value, numbers.Number):
                    dim = 1
                else:
                    dim = []
                bias = RootVariable(0.,
                                    self.name + "_" + parameter_name + "_" + "bias",
                                    learnable, is_observed=self._observed)
                mixing = RootVariable(5.,
                                      self.name + "_" + parameter_name + "_" + "mixing",
                                      learnable, is_observed=self._observed)
                kwargs.update({parameter_name: ranges[parameter_name].forward_transform(BF.sigmoid(mixing)*value + (1 - BF.sigmoid(mixing))*bias, dim)})

    def _check_for_stochastic_process_arguments(self):
        """
        Private method. Looks in the input variables to see if an input is a Process. If there is a process this
        variable should also be a process. Throws an ValueError is there are more then one processes in the input.

        Returns:
            None.
        """
        params = self._input
        process_params = [key for key, argument in params.items()
                          if isinstance(argument, Process)]
        if len(process_params) > 1:
            raise ValueError("A random variable can only have one stochastic process as input parameter")
        return len(process_params) == 1

    def _construct_variable_or_process(self):
        """
        Private method. Checks if this variable should be a process or a standard variable. If it should be a process
        a new process is constructed from the input variables.

        Returns:
            brancher.StandardVariable or brancher.Process.
        """
        if self._check_for_stochastic_process_arguments():
            params = self._input
            process_params = [key for key, argument in params.items()
                              if isinstance(argument, Process)]
            param_name = process_params[0]
            process = params.pop(param_name)
            variable_constructor = self.__class__
            observation_cond_dist = lambda x: variable_constructor(**{**params, **{param_name: x}}, name=self.name)
            process.attach_observation_model(observation_cond_dist)
            return process
        else:
            return self


class EmpiricalVariable(StandardVariable):
    """
    Variable that represents samples from a dataset.

    Parameters
    ----------
    dataset: np.ndarray or torch.Tensor. Dataset from which new samples come.

    batchsize: number. The number of items from the dataset that are in each sample. If not given, batchsize if number
    of indices.

    indices: 1D np.array or List. The items of the dataset to return when sampled. If not given, samples are randomly
    chosen.

    weights: 1D np.array or List. Probabilities associated to each item in the dataset. If not given, items are
    uniformly chosen.
    """
    def __init__(self, dataset, name, learnable=False, has_bias=False, is_observed=False, batch_size=None, indices=None, weights=None): #TODO: Ugly logic
        self._type = "Empirical"
        input_parameters = {"dataset": dataset, "batch_size": batch_size, "indices": indices, "weights": weights}
        ranges = {par_name: geometric_ranges.UnboundedRange()
                  for par_name, par_value in input_parameters.items()
                  if par_value is not None}
        kwargs = {par_name: par_value
                  for par_name, par_value in input_parameters.items()
                  if par_value is not None}
        super().__init__(name, **kwargs, learnable=learnable, has_bias=has_bias, ranges=ranges, is_observed=is_observed)

        if not batch_size:
            if indices:
                batch_size = len(indices)
            else:
                raise ValueError("Either the indices or the batch size has to be given as input")

        self.batch_size = batch_size
        self.distribution = distributions.EmpiricalDistribution(batch_size=batch_size, is_observed=is_observed)


class RandomIndices(EmpiricalVariable):
    """
    Variable that represents random indices from a dataset. This can be used for random sampling from empirical variables.

    Parameters
    ----------
    dataset_size: Number. Size of the dataset from which random indices will be sampled. Indices will be in the
    range 0 to dataset_size.

    batch_size: Number. Number of indices to generate per sample.
    """
    def __init__(self, dataset_size, batch_size, name, has_bias=False, is_observed=False):
        self._type = "Random Index"
        super().__init__(dataset=list(range(dataset_size)),
                         batch_size=batch_size, has_bias=has_bias, is_observed=is_observed, name=name)

    def __len__(self):
        return self.batch_size


class DeterministicVariable(StandardVariable): #TODO: Future refactor? Should Deterministic variables and deterministic node be different? (No probably not)
    """
    Variable that represents a deterministic value.

    Parameters
    ----------
    value: brancher.Variables, numbers, numpy.ndarrays, torch.Tensors, or List/Tuple of
    brancher.Variables and brancher.PartialLinks. The value of this variable.
    log_determinant: PartialLink. The log determinant of the transformation used to obtain the value of this variable.
    """
    def __init__(self, value, name, log_determinant=None, learnable=False, has_bias=False, is_observed=False, variable_range=geometric_ranges.UnboundedRange()):
        self._type = "Deterministic node"
        if not isinstance(log_determinant, PartialLink):
            if log_determinant is None:
                log_determinant = torch.tensor(np.zeros((1, 1))).float().to(device)
            var2link(log_determinant)
        ranges = {"value": variable_range,
                  "log_determinant": geometric_ranges.UnboundedRange()}
        super().__init__(name, value=value, log_determinant=log_determinant, learnable=learnable, has_bias=has_bias, ranges=ranges, is_observed=is_observed)
        self.distribution = distributions.DeterministicDistribution()

    @property
    def value(self):
        return self._get_sample(1)[self]


class NormalVariable(StandardVariable):
    """
    A normally distributed variable.

    Parameters
    ----------
    loc: brancher.Variables, numbers, numpy.ndarrays, torch.Tensors, or List/Tuple of
    brancher.Variables and brancher.PartialLinks. The location or mean or expected value of the normal variable.

    scale: brancher.Variables, numbers, numpy.ndarrays, torch.Tensors, or List/Tuple of
    brancher.Variables and brancher.PartialLinks. The scale or standard deviation of the normal variable.
    """
    def __init__(self, loc, scale, name, learnable=False, has_bias=False, is_observed=False):
        self._type = "Normal"
        ranges = {"loc": geometric_ranges.UnboundedRange(),
                  "scale": geometric_ranges.RightHalfLine(0.)}
        super().__init__(name, loc=loc, scale=scale, learnable=learnable, has_bias=has_bias,
                         ranges=ranges, is_observed=is_observed)
        self.distribution = distributions.NormalDistribution()


class StudentTVariable(StandardVariable):
    """
    A variable with a Student's t-distribution.

    Parameters
    ----------
    loc: brancher.Variables, numbers, numpy.ndarrays, torch.Tensors, or List/Tuple of
    brancher.Variables and brancher.PartialLinks. The location or mean of the Student-t variable.

    scale: brancher.Variables, numbers, numpy.ndarrays, torch.Tensors, or List/Tuple of
    brancher.Variables and brancher.PartialLinks. The scale of the Student-t variable.

    df: brancher.Variables, numbers, numpy.ndarrays, torch.Tensors, or List/Tuple of
    brancher.Variables and brancher.PartialLinks. The degrees of freedom of a Student-t variable.
    """
    def __init__(self, df, loc, scale, name, learnable=False, has_bias=False, is_observed=False):
        self._type = "StudentT"
        ranges = {"df": geometric_ranges.UnboundedRange(),
                  "loc": geometric_ranges.UnboundedRange(),
                  "scale": geometric_ranges.RightHalfLine(0.)}
        super().__init__(name, df=df, loc=loc, scale=scale, learnable=learnable, has_bias=has_bias,
                         ranges=ranges, is_observed=is_observed)
        self.distribution = distributions.StudentTDistribution()


class UniformVariable(StandardVariable):
    """
    A variable with a uniform distribution.

    Parameters
    ----------
    low: brancher.Variables, numbers, numpy.ndarrays, torch.Tensors, or List/Tuple of
    brancher.Variables and brancher.PartialLinks. Minimum value of the uniform variable.

    high: brancher.Variables, numbers, numpy.ndarrays, torch.Tensors, or List/Tuple of
    brancher.Variables and brancher.PartialLinks. Maximum value of the uniform variable.
    """
    def __init__(self, low, high, name, learnable=False, has_bias=False, is_observed=False):
        self._type = "Uniform"
        ranges = {"low": geometric_ranges.UnboundedRange(),
                  "high": geometric_ranges.UnboundedRange()}
        super().__init__(name, low=low, high=high, learnable=learnable,
                         has_bias=has_bias, ranges=ranges, is_observed=is_observed)
        self.distribution = distributions.UniformDistribution()


class CauchyVariable(StandardVariable):
    """
    A variable with a Cauchy distribution.

    Parameters
    ----------
    loc: brancher.Variables, numbers, numpy.ndarrays, torch.Tensors, or List/Tuple of
    brancher.Variables and brancher.PartialLinks. The mode or median of the Cauchy variable.

    scale: brancher.Variables, numbers, numpy.ndarrays, torch.Tensors, or List/Tuple of
    brancher.Variables and brancher.PartialLinks. The scale determines the half width at half maximum of the Cauchy variable.
    """
    def __init__(self, loc, scale, name, learnable=False, has_bias=False, is_observed=False):
        self._type = "Cauchy"
        ranges = {"loc": geometric_ranges.UnboundedRange(),
                  "scale": geometric_ranges.RightHalfLine(0.)}
        super().__init__(name, loc=loc, scale=scale, learnable=learnable,
                         has_bias=has_bias, ranges=ranges, is_observed=is_observed)
        self.distribution = distributions.CauchyDistribution()


class HalfCauchyVariable(StandardVariable):
    """
    A variable with a half Cauchy distribution.

    Parameters
    ----------
    scale: brancher.Variables, numbers, numpy.ndarrays, torch.Tensors, or List/Tuple of
    brancher.Variables and brancher.PartialLinks. The scale of the full Cauchy distribution.
    """
    def __init__(self, scale, name, learnable=False, has_bias=False, is_observed=False):
        self._type = "HalfCauchy"
        ranges = {"scale": geometric_ranges.RightHalfLine(0.)}
        super().__init__(name, scale=scale, learnable=learnable,
                         has_bias=has_bias, ranges=ranges, is_observed=is_observed)
        self.distribution = distributions.HalfCauchyDistribution()


class HalfNormalVariable(StandardVariable):
    """
    A variable with a half-normal distribution.

    Parameters
    ----------
    scale: brancher.Variables, numbers, numpy.ndarrays, torch.Tensors, or List/Tuple of
    brancher.Variables and brancher.PartialLinks. The scale of the full normal distribution.
    """
    def __init__(self, scale, name, learnable=False, has_bias=False, is_observed=False):
        self._type = "HalfNormal"
        ranges = {"scale": geometric_ranges.RightHalfLine(0.)}
        super().__init__(name, scale=scale, learnable=learnable,
                         has_bias=has_bias, ranges=ranges, is_observed=is_observed)
        self.distribution = distributions.HalfNormalDistribution()


class Chi2Variable(StandardVariable):
    """
    A variable with a Chi^2 distribution.

    Parameters
    ----------
    df: brancher.Variables, numbers, numpy.ndarrays, torch.Tensors, or List/Tuple of
    brancher.Variables and brancher.PartialLinks. The degrees of freedom of a Chi^2 variable.
    """
    def __init__(self, df, name, learnable=False, has_bias=False, is_observed=False):
        self._type = "Chi2"
        ranges = {"df": geometric_ranges.UnboundedRange()} #TODO: Natural number
        super().__init__(name, df=df, learnable=learnable,
                         has_bias=has_bias, ranges=ranges, is_observed=is_observed)
        self.distribution = distributions.Chi2Distribution()


class GumbelVariable(StandardVariable):
    """
    A variable with a Gumbel distribution.

    Parameters
    ----------
    loc: brancher.Variables, numbers, numpy.ndarrays, torch.Tensors, or List/Tuple of
    brancher.Variables and brancher.PartialLinks. The location of the Gumbel variable.

    scale: brancher.Variables, numbers, numpy.ndarrays, torch.Tensors, or List/Tuple of
    brancher.Variables and brancher.PartialLinks. The scale of the Gumbel variable.
    """
    def __init__(self, loc, scale, name, learnable=False, has_bias=False, is_observed=False):
        self._type = "Gumbel"
        ranges = {"loc": geometric_ranges.UnboundedRange(),
                  "scale": geometric_ranges.RightHalfLine(0.)}
        super().__init__(name, loc=loc, scale=scale, learnable=learnable,
                         has_bias=has_bias, ranges=ranges, is_observed=is_observed)
        self.distribution = distributions.GumbelDistribution()


class LaplaceVariable(StandardVariable):
    """
    A variable with a Laplace distribution.

    Parameters
    ----------
    loc: brancher.Variables, numbers, numpy.ndarrays, torch.Tensors, or List/Tuple of
    brancher.Variables and brancher.PartialLinks. The location of the Laplace variable.

    scale: brancher.Variables, numbers, numpy.ndarrays, torch.Tensors, or List/Tuple of
    brancher.Variables and brancher.PartialLinks. The scale of the Laplace variable.
    """
    def __init__(self, loc, scale, name, learnable=False, has_bias=False, is_observed=False):
        self._type = "Laplace"
        ranges = {"loc": geometric_ranges.UnboundedRange(),
                  "scale": geometric_ranges.RightHalfLine(0.)}
        super().__init__(name, loc=loc, scale=scale, learnable=learnable,
                         has_bias=has_bias, ranges=ranges, is_observed=is_observed)
        self.distribution = distributions.LaplaceDistribution()


class ExponentialVariable(StandardVariable):
    """
    A variable with a exponential distribution.

    Parameters
    ----------
    rate: brancher.Variables, numbers, numpy.ndarrays, torch.Tensors, or List/Tuple of
    brancher.Variables and brancher.PartialLinks. The rate or inverse scale of the exponential variable.
    """
    def __init__(self, rate, name, learnable=False, has_bias=False, is_observed=False):
        self._type = "Exponential"
        ranges = {"rate": geometric_ranges.RightHalfLine(0.)}
        super().__init__(name, rate=rate, learnable=learnable,
                         has_bias=has_bias, ranges=ranges, is_observed=is_observed)
        self.distribution = distributions.ExponentialDistribution()


class PoissonVariable(StandardVariable):
    """
    A variable with a Poisson distribution.

    Parameters
    ----------
    rate: brancher.Variables, numbers, numpy.ndarrays, torch.Tensors, or List/Tuple of
    brancher.Variables and brancher.PartialLinks. The rate of the Poisson variable.
    """
    def __init__(self, rate, name, learnable=False, has_bias=False, is_observed=False):
        self._type = "Poisson"
        ranges = {"rate": geometric_ranges.RightHalfLine(0.)}
        super().__init__(name, rate=rate, learnable=learnable,
                         has_bias=has_bias, ranges=ranges, is_observed=is_observed)
        self.distribution = distributions.PoissonDistribution()


class LogNormalVariable(StandardVariable):
    """
    A variable with a log-normal distribution.

    Parameters
    ----------
    loc: brancher.Variables, numbers, numpy.ndarrays, torch.Tensors, or List/Tuple of
    brancher.Variables and brancher.PartialLinks. The mean of the log-normal variable.

    scale: brancher.Variables, numbers, numpy.ndarrays, torch.Tensors, or List/Tuple of
    brancher.Variables and brancher.PartialLinks. The standard deviation of the log-normal variable.
    """
    def __init__(self, loc, scale, name, learnable=False, has_bias=False, is_observed=False):
        self._type = "Log Normal"
        ranges = {"loc": geometric_ranges.UnboundedRange(),
                  "scale": geometric_ranges.RightHalfLine(0.)}
        super().__init__(name, loc=loc, scale=scale, learnable=learnable,
                         has_bias=has_bias, ranges=ranges, is_observed=is_observed)
        self.distribution = distributions.LogNormalDistribution()


#class LogitNormalVariable(VariableConstructor):
#    """
#    Summary
#
#    Parameters
#    ----------
#    """
#    def __init__(self, loc, scale, name, learnable=False, is_observed=False):
#        self._type = "Logit Normal"
#        ranges = {"loc": geometric_ranges.UnboundedRange(),
#                  "scale": geometric_ranges.RightHalfLine(0.)}
#        super().__init__(name, loc=loc, scale=scale, learnable=learnable, ranges=ranges, is_observed=is_observed)
#        self.distribution = distributions.LogitNormalDistribution()


class BetaVariable(StandardVariable):
    """
    A variable with a beta distribution.

    Parameters
    ----------
    alpha: brancher.Variables, numbers, numpy.ndarrays, torch.Tensors, or List/Tuple of
    brancher.Variables and brancher.PartialLinks. The alpha parameter of the beta distribution.

    beta: brancher.Variables, numbers, numpy.ndarrays, torch.Tensors, or List/Tuple of
    brancher.Variables and brancher.PartialLinks. The beta parameter of the beta distribution.
    """
    def __init__(self, alpha, beta, name, learnable=False, has_bias=False, is_observed=False):
        self._type = "Beta"
        concentration1 = alpha
        concentration0 = beta
        ranges = {"concentration1": geometric_ranges.RightHalfLine(0.),
                  "concentration0": geometric_ranges.RightHalfLine(0.)}
        super().__init__(name, concentration1=concentration1, concentration0=concentration0,
                         learnable=learnable, has_bias=has_bias, ranges=ranges, is_observed=is_observed)
        self.distribution = distributions.BetaDistribution()


class BinomialVariable(StandardVariable):
    """
    A variable with a binomial distribution that represents a number of successes in a total_count number of
    experiments. The chance of success is given with probability or logits.

    Parameters
    ----------
    total_count: brancher.Variables, numbers, numpy.ndarrays, torch.Tensors, or List/Tuple of
    brancher.Variables and brancher.PartialLinks. The number of experiments or trails with outcome success or failure.

    probs: brancher.Variables, numbers, numpy.ndarrays, torch.Tensors, or List/Tuple of
    brancher.Variables and brancher.PartialLinks. The probability of sampling a success.

    logits: alpha: brancher.Variables, numbers, numpy.ndarrays, torch.Tensors, or List/Tuple of
    brancher.Variables and brancher.PartialLinks. The log-odds of sampling a success.
    """
    def __init__(self, total_count, probs=None, logits=None, name="Binomial", learnable=False, has_bias=False, is_observed=False):
        self._type = "Binomial"
        if probs is not None and logits is None:
            ranges = {"total_count": geometric_ranges.UnboundedRange(), #TODO: It should become natural number in the future
                      "probs": geometric_ranges.Interval(0., 1.)}
            super().__init__(name, total_count=total_count, probs=probs, learnable=learnable,
                             has_bias=has_bias, ranges=ranges, is_observed=is_observed)
            self.distribution = distributions.BinomialDistribution()
        elif logits is not None and probs is None:
            ranges = {"total_count": geometric_ranges.UnboundedRange(),
                      "logits": geometric_ranges.UnboundedRange()}
            super().__init__(name, total_count=total_count, logits=logits, learnable=learnable,
                             has_bias=has_bias, ranges=ranges)
            self.distribution = distributions.BinomialDistribution()
        else:
            raise ValueError("Either probs or " +
                             "logits needs to be provided as input")


class NegativeBinomialVariable(StandardVariable):
    """
    A variable with a negative binomial distribution that represents a number of success in a experiment until a
    total_count number of failures has occured. The chance of success is given with probability or logits.

    Parameters
    ----------
    total_count: brancher.Variables, numbers, numpy.ndarrays, torch.Tensors, or List/Tuple of
    brancher.Variables and brancher.PartialLinks. The number of experiments with outcome failure.

    probs: brancher.Variables, numbers, numpy.ndarrays, torch.Tensors, or List/Tuple of
    brancher.Variables and brancher.PartialLinks. The probability of sampling a success.

    logits: alpha: brancher.Variables, numbers, numpy.ndarrays, torch.Tensors, or List/Tuple of
    brancher.Variables and brancher.PartialLinks. The log-odds of sampling a success.
    """
    def __init__(self, total_count, probs=None, logits=None, name="NegativeBinomial", learnable=False, has_bias=False, is_observed=False):
        self._type = "NegativeBinomial"
        if probs is not None and logits is None:
            ranges = {"total_count": geometric_ranges.UnboundedRange(), #TODO: It should become natural number in the future
                      "probs": geometric_ranges.Interval(0., 1.)}
            super().__init__(name, total_count=total_count, probs=probs, learnable=learnable,
                             has_bias=has_bias, ranges=ranges, is_observed=is_observed)
            self.distribution = distributions.BinomialDistribution()
        elif logits is not None and probs is None:
            ranges = {"total_count": geometric_ranges.UnboundedRange(),
                      "logits": geometric_ranges.UnboundedRange()}
            super().__init__(name, total_count=total_count, logits=logits, learnable=learnable,
                             has_bias=has_bias, ranges=ranges)
            self.distribution = distributions.NegativeBinomialDistribution()
        else:
            raise ValueError("Either probs or " +
                             "logits needs to be provided as input")


class BernoulliVariable(StandardVariable):
    """
    A variable with a Bernoulli distribution that represents the result of a success/failure experiment. The chance of
    success is given with probability or logits.

    Parameters
    ----------
    probs: brancher.Variables, numbers, numpy.ndarrays, torch.Tensors, or List/Tuple of
    brancher.Variables and brancher.PartialLinks. The probability of sampling a success.

    logits: alpha: brancher.Variables, numbers, numpy.ndarrays, torch.Tensors, or List/Tuple of
    brancher.Variables and brancher.PartialLinks. The log-odds of sampling a success.
    """
    def __init__(self, probs=None, logits=None, name="Bernoulli", learnable=False, has_bias=False, is_observed=False):
        self._type = "Bernoulli"
        if probs is not None and logits is None:
            ranges = {"probs": geometric_ranges.Interval(0., 1.)}
            super().__init__(name, probs=probs, learnable=learnable,
                             has_bias=has_bias, ranges=ranges, is_observed=is_observed)
            self.distribution = distributions.BernoulliDistribution()
        elif logits is not None and probs is None:
            ranges = {"logits": geometric_ranges.UnboundedRange()}
            super().__init__(name, logits=logits, learnable=learnable,
                             has_bias=has_bias, ranges=ranges)
            self.distribution = distributions.BernoulliDistribution()
        else:
            raise ValueError("Either probs or " +
                             "logits needs to be provided as input")


class GeometricVariable(StandardVariable):
    """
    A variable with a geometric distribution that represents the number of success/failure experiments that needed to be
    done until a success outcome. The chance of success is given with probability or logits.

    Parameters
    ----------
    probs: brancher.Variables, numbers, numpy.ndarrays, torch.Tensors, or List/Tuple of
    brancher.Variables and brancher.PartialLinks. The probability of sampling a success.

    logits: alpha: brancher.Variables, numbers, numpy.ndarrays, torch.Tensors, or List/Tuple of
    brancher.Variables and brancher.PartialLinks. The log-odds of sampling a success.
    """
    def __init__(self, probs=None, logits=None, name="Geometric", learnable=False, has_bias=False, is_observed=False):
        self._type = "Geometric"
        if probs is not None and logits is None:
            ranges = {"probs": geometric_ranges.Interval(0., 1.)}
            super().__init__(name, probs=probs, learnable=learnable,
                             has_bias=has_bias, ranges=ranges, is_observed=is_observed)
            self.distribution = distributions.GeometricDistribution()
        elif logits is not None and probs is None:
            ranges = {"logits": geometric_ranges.UnboundedRange()}
            super().__init__(name, logits=logits, learnable=learnable,
                             has_bias=has_bias, ranges=ranges)
            self.distribution = distributions.GeometricDistribution()
        else:
            raise ValueError("Either probs or " +
                             "logits needs to be provided as input")


class CategoricalVariable(StandardVariable):
    """
    A variable with a categorical distribution where the probabilities or logits need to be specified per category.

    Parameters
    ----------
    probs: brancher.Variables, numbers, numpy.ndarrays, torch.Tensors, or List/Tuple of
    brancher.Variables and brancher.PartialLinks. The probabilities of sampling a category.

    logits: alpha: brancher.Variables, numbers, numpy.ndarrays, torch.Tensors, or List/Tuple of
    brancher.Variables and brancher.PartialLinks. The log-odds of sampling a category.
    """
    def __init__(self, probs=None, logits=None, name="Categorical", learnable=False, has_bias=False, is_observed=False):
        self._type = "Categorical"
        if probs is not None and logits is None:
            ranges = {"p": geometric_ranges.Simplex()}
            super().__init__(name, probs=probs, learnable=learnable,
                             has_bias=has_bias, ranges=ranges, is_observed=is_observed)
            self.distribution = distributions.CategoricalDistribution()
        elif logits is not None and probs is None:
            ranges = {"logits": geometric_ranges.UnboundedRange()}
            super().__init__(name, logits=logits, learnable=learnable,
                             has_bias=has_bias, ranges=ranges, is_observed=is_observed)
            self.distribution = distributions.CategoricalDistribution()
        else:
            raise ValueError("Either probs or " +
                             "logits needs to be provided as input")


#class ConcreteVariable(VariableConstructor):
#    """
#    Summary
#
#    Parameters
#    ----------
#    """
#    def __init__(self, tau, probs, name, learnable=False, is_observed=False):
#        self._type = "Concrete"
#        ranges = {"tau": geometric_ranges.RightHalfLine(0.),
#                  "p": geometric_ranges.Simplex()}
#        super().__init__(name, tau=tau, probs=probs, learnable=learnable, ranges=ranges, is_observed=is_observed)
#        self.distribution = distributions.ConcreteDistribution()


class MultivariateNormalVariable(StandardVariable):
    """
    A variable with a multivariate normal or gaussian distribution specified by a mean vector and either a positive
    definite covariance matrix or a positive definite precision matrix or a lower-triangular matrix with a
    positive-valued diagonal.

    Parameters
    ----------
    loc: brancher.Variables, numbers, numpy.ndarrays, torch.Tensors, or List/Tuple of
    brancher.Variables and brancher.PartialLinks. The location or mean vector of the multivariate-normal variable.

    covariance_matrix: brancher.Variables, numbers, numpy.ndarrays, torch.Tensors, or List/Tuple of
    brancher.Variables and brancher.PartialLinks. The positive-definite covariance matrix.

    precision_matrix: brancher.Variables, numbers, numpy.ndarrays, torch.Tensors, or List/Tuple of
    brancher.Variables and brancher.PartialLinks. The positive-definite precision matrix.

    scale_tril: brancher.Variables, numbers, numpy.ndarrays, torch.Tensors, or List/Tuple of
    brancher.Variables and brancher.PartialLinks. The lower-triangular factor of covariance, with positive-valued diagonal.
    """
    def __init__(self, loc, covariance_matrix=None, precision_matrix=None,
                 scale_tril=None, name="Multivariate Normal", learnable=False, has_bias=False, is_observed=False):
        self._type = "Multivariate Normal"
        if scale_tril is not None and covariance_matrix is None and precision_matrix is None:
            ranges = {"loc": geometric_ranges.UnboundedRange(),
                      "scale_tril": geometric_ranges.UnboundedRange()}
            super().__init__(name, loc=loc, scale_tril=scale_tril, learnable=learnable,
                             has_bias=has_bias, ranges=ranges, is_observed=is_observed)
            self.distribution = distributions.MultivariateNormalDistribution()

        elif scale_tril is None and covariance_matrix is not None and precision_matrix is None:
            ranges = {"loc": geometric_ranges.UnboundedRange(),
                      "covariance_matrix": geometric_ranges.PositiveDefiniteMatrix()}
            super().__init__(name, loc=loc, covariance_matrix=covariance_matrix, learnable=learnable,
                             has_bias=has_bias, ranges=ranges, is_observed=is_observed)
            self.distribution = distributions.MultivariateNormalDistribution()

        elif scale_tril is None and covariance_matrix is None and precision_matrix is not None:
            ranges = {"loc": geometric_ranges.UnboundedRange(),
                      "precision_matrix": geometric_ranges.PositiveDefiniteMatrix()}
            super().__init__(name, loc=loc, precision_matrix=precision_matrix, learnable=learnable,
                             has_bias=has_bias, ranges=ranges, is_observed=is_observed)
            self.distribution = distributions.MultivariateNormalDistribution()

        else:
            raise ValueError("Either covariance_matrix or precision_matrix or"+
                             "scale_tril needs to be provided as input")


class DirichletVariable(StandardVariable):
    """
    A variable with a Dirichlet distribution.

    Parameters
    ----------
    concentration: brancher.Variables, numbers, numpy.ndarrays, torch.Tensors, or List/Tuple of
    brancher.Variables and brancher.PartialLinks. The concentration parameter of the Dirichlet variable.
    """
    def __init__(self, concentration, name, learnable=False, has_bias=False, is_observed=False):
        self._type = "Dirichlet"
        ranges = {"concentration": geometric_ranges.RightHalfLine(0.)}
        super().__init__(name, concentration=concentration,
                         learnable=learnable,
                         has_bias=has_bias, ranges=ranges, is_observed=is_observed)
        self.distribution = distributions.DirichletDistribution()