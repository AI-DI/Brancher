import numbers

import numpy as np
import torch.nn as nn

import brancher.distributions as distributions
import brancher.functions as BF
import brancher.geometric_ranges as geometric_ranges
from brancher.variables import var2link, Variable, RootVariable, RandomVariable, PartialLink
from brancher.utilities import join_sets_list



class LinkConstructor(nn.ModuleList):
    """
    Summary

    Parameters
    ----------
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        modules = [link
                 for partial_link in kwargs.values()
                 for link in var2link(partial_link).links]
        super().__init__(modules) #TODO: asserts that specified links are valid pytorch modules

    def __call__(self, values):
        return {k: var2link(x).fn(values) for k, x in self.kwargs.items()}


class VariableConstructor(RandomVariable):
    """
    Summary

    Parameters
    ----------
    """
    def __init__(self, name, learnable, ranges, is_observed=False, **kwargs): #TODO: code duplication here
        self.name = name
        self._evaluated = False
        self._observed = is_observed
        self._observed_value = None
        self._current_value = None
        self.construct_deterministic_parents(learnable, ranges, kwargs)
        self.parents = join_sets_list([var2link(x).vars for x in kwargs.values()])
        self.ancestors = join_sets_list([self.parents] + [parent.ancestors for parent in self.parents])
        self.link = LinkConstructor(**kwargs)
        self.samples = None
        self.ranges = {}
        self.dataset = None
        self.has_random_dataset = False
        self.has_observed_value = False
        self.is_normalized = True
        self.partial_links = {name: var2link(link) for name, link in kwargs.items()}

    def construct_deterministic_parents(self, learnable, ranges, kwargs):
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


class EmpiricalVariable(VariableConstructor):
    """
    Summary

    Parameters
    ----------
    """
    def __init__(self, dataset, name, learnable=False, is_observed=False, batch_size=None, indices=None, weights=None): #TODO: Ugly logic
        self._type = "Empirical"
        input_parameters = {"dataset": dataset, "batch_size": batch_size, "indices": indices, "weights": weights}
        ranges = {par_name: geometric_ranges.UnboundedRange()
                  for par_name, par_value in input_parameters.items()
                  if par_value is not None}
        kwargs = {par_name: par_value
                  for par_name, par_value in input_parameters.items()
                  if par_value is not None}
        super().__init__(name, **kwargs, learnable=learnable, ranges=ranges, is_observed=is_observed)

        if not batch_size:
            if indices:
                batch_size = len(indices)
            else:
                raise ValueError("Either the indices or the batch size has to be given as input")

        self.batch_size = batch_size
        self.distribution = distributions.EmpiricalDistribution(batch_size=batch_size, is_observed=is_observed)


class RandomIndices(EmpiricalVariable):
    """
    Summary

    Parameters
    ----------
    """
    def __init__(self, dataset_size, batch_size, name, is_observed=False):
        self._type = "Random Index"
        super().__init__(dataset=list(range(dataset_size)),
                         batch_size=batch_size, is_observed=is_observed, name=name)

    def __len__(self):
        return self.batch_size


class DeterministicVariable(VariableConstructor): #TODO: Future refactor? Should Deterministic variables and deterministic node be different? (No probably not)
    """
    Summary

    Parameters
    ----------
    """
    def __init__(self, value, name, learnable=False, is_observed=False, variable_range=geometric_ranges.UnboundedRange()):
        self._type = "Deterministic node"
        ranges = {"value": variable_range}
        super().__init__(name, value=value, learnable=learnable, ranges=ranges, is_observed=is_observed)
        self.distribution = distributions.DeterministicDistribution()

    @property
    def value(self):
        return self._get_sample(1)[self]


class NormalVariable(VariableConstructor):
    """
    Summary

    Parameters
    ----------
    """
    def __init__(self, loc, scale, name, learnable=False, is_observed=False):
        self._type = "Normal"
        ranges = {"loc": geometric_ranges.UnboundedRange(),
                  "scale": geometric_ranges.RightHalfLine(0.)}
        super().__init__(name, loc=loc, scale=scale, learnable=learnable, ranges=ranges, is_observed=is_observed)
        self.distribution = distributions.NormalDistribution()

    #def __add__(self, other):
    #    if isinstance(other, NormalVariable):
    #        return NormalVariable(self.partial_links["loc"] + other.partial_links["loc"],
    #                              scale=BF.sqrt(self.partial_links["scale"]**2 + other.partial_links["scale"]**2),
    #                              name=self.name + " + " + other.name, learnable=False)
    #    else:
    #        return super().__add__(other)


class CauchyVariable(VariableConstructor):
    """
    Summary

    Parameters
    ----------
    """
    def __init__(self, loc, scale, name, learnable=False, is_observed=False):
        self._type = "Cauchy"
        ranges = {"loc": geometric_ranges.UnboundedRange(),
                  "scale": geometric_ranges.RightHalfLine(0.)}
        super().__init__(name, loc=loc, scale=scale, learnable=learnable, ranges=ranges, is_observed=is_observed)
        self.distribution = distributions.CauchyDistribution()


class LaplaceVariable(VariableConstructor):
    """
    Summary

    Parameters
    ----------
    """
    def __init__(self, loc, scale, name, learnable=False, is_observed=False):
        self._type = "Laplace"
        ranges = {"loc": geometric_ranges.UnboundedRange(),
                  "scale": geometric_ranges.RightHalfLine(0.)}
        super().__init__(name, loc=loc, scale=scale, learnable=learnable, ranges=ranges, is_observed=is_observed)
        self.distribution = distributions.LaplaceDistribution()


class LogNormalVariable(VariableConstructor):
    """
    Summary

    Parameters
    ----------
    """
    def __init__(self, loc, scale, name, learnable=False, is_observed=False):
        self._type = "Log Normal"
        ranges = {"loc": geometric_ranges.UnboundedRange(),
                  "scale": geometric_ranges.RightHalfLine(0.)}
        super().__init__(name, loc=loc, scale=scale, learnable=learnable, ranges=ranges, is_observed=is_observed)
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


class BetaVariable(VariableConstructor):
    """
    Summary

    Parameters
    ----------
    """
    def __init__(self, alpha, beta, name, learnable=False, is_observed=False):
        self._type = "Logit Normal"
        concentration1 = alpha
        concentration0 = beta
        ranges = {"concentration1": geometric_ranges.RightHalfLine(0.),
                  "concentration0": geometric_ranges.RightHalfLine(0.)}
        super().__init__(name, concentration1=concentration1, concentration0=concentration0,
                         learnable=learnable, ranges=ranges, is_observed=is_observed)
        self.distribution = distributions.BetaDistribution()


class BinomialVariable(VariableConstructor):
    """
    Summary

    Parameters
    ----------
    """
    def __init__(self, total_count, probs=None, logits=None, name="Binomial", learnable=False, is_observed=False):
        self._type = "Binomial"
        if probs is not None and logits is None:
            ranges = {"total_count": geometric_ranges.UnboundedRange(),
                      "probs": geometric_ranges.Interval(0., 1.)}
            super().__init__(name, total_count=total_count, probs=probs, learnable=learnable, ranges=ranges, is_observed=is_observed)
            self.distribution = distributions.BinomialDistribution()
        elif logits is not None and probs is None:
            ranges = {"total_count": geometric_ranges.UnboundedRange(),
                      "logits": geometric_ranges.UnboundedRange()}
            super().__init__(name, total_count=total_count, logits=logits, learnable=learnable, ranges=ranges)
            self.distribution = distributions.BinomialDistribution()
        else:
            raise ValueError("Either probs or " +
                             "logits needs to be provided as input")


class BernulliVariable(VariableConstructor):
    """
    Summary

    Parameters
    ----------
    """
    def __init__(self, probs=None, logits=None, name="Bernulli", learnable=False, is_observed=False):
        self._type = "Bernulli"
        if probs is not None and logits is None:
            ranges = {"probs": geometric_ranges.Interval(0., 1.)}
            super().__init__(name, probs=probs, learnable=learnable, ranges=ranges, is_observed=is_observed)
            self.distribution = distributions.BernulliDistribution()
        elif logits is not None and probs is None:
            ranges = {"logits": geometric_ranges.UnboundedRange()}
            super().__init__(name, logits=logits, learnable=learnable, ranges=ranges)
            self.distribution = distributions.BernulliDistribution()
        else:
            raise ValueError("Either probs or " +
                             "logits needs to be provided as input")


class CategoricalVariable(VariableConstructor): #TODO: Work in progress
    """
    Summary

    Parameters
    ----------
    """
    def __init__(self, probs=None, logits=None, name="Categorical", learnable=False, is_observed=False):
        self._type = "Categorical"
        if probs is not None and logits is None:
            ranges = {"p": geometric_ranges.Simplex()}
            super().__init__(name, probs=probs, learnable=learnable, ranges=ranges, is_observed=is_observed)
            self.distribution = distributions.CategoricalDistribution()
        elif logits is not None and probs is None:
            ranges = {"logits": geometric_ranges.UnboundedRange()}
            super().__init__(name, logits=logits, learnable=learnable, ranges=ranges, is_observed=is_observed)
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


class MultivariateNormalVariable(VariableConstructor):
    """
    Summary

    Parameters
    ----------
    """
    def __init__(self, loc, covariance_matrix=None, precision_matrix=None,
                 scale_tril=None, name="Multivariate Normal", learnable=False, is_observed=False):
        self._type = "Multivariate Normal"
        if scale_tril is not None and covariance_matrix is None and precision_matrix is None:
            ranges = {"loc": geometric_ranges.UnboundedRange(),
                      "scale_tril": geometric_ranges.UnboundedRange()}
            super().__init__(name, loc=loc, scale_tril=scale_tril, learnable=learnable, ranges=ranges, is_observed=is_observed)
            self.distribution = distributions.MultivariateNormalDistribution()

        elif scale_tril is None and covariance_matrix is not None and precision_matrix is None:
            ranges = {"loc": geometric_ranges.UnboundedRange(),
                      "covariance_matrix": geometric_ranges.PositiveDefiniteMatrix()}
            super().__init__(name, loc=loc, covariance_matrix=covariance_matrix, learnable=learnable, ranges=ranges, is_observed=is_observed)
            self.distribution = distributions.MultivariateNormalDistribution()

        elif scale_tril is None and covariance_matrix is None and precision_matrix is not None:
            ranges = {"loc": geometric_ranges.UnboundedRange(),
                      "precision_matrix": geometric_ranges.UnboundedRange()}
            super().__init__(name, loc=loc, precision_matrix=precision_matrix, learnable=learnable, ranges=ranges, is_observed=is_observed)
            self.distribution = distributions.MultivariateNormalDistribution()

        else:
            raise ValueError("Either covariance_matrix or precision_matrix or"+
                             "scale_tril needs to be provided as input")