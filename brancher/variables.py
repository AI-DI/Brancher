"""
Variables
---------
The variables module contains base classes for defining random and deterministic variables and the operations that can
be done on these variables. It also contains the base classes of probabilistic models which is a collection of
deterministic and random variables.
"""
from abc import ABC, abstractmethod
import operator
import numbers
import collections
from collections.abc import Iterable

from brancher.modules import ParameterModule

import numpy as np
import pandas as pd
import torch

import warnings

from brancher import distributions
from brancher import gradient_estimators

from brancher.utilities import join_dicts_list, join_sets_list
from brancher.utilities import flatten_list
from brancher.utilities import partial_broadcast
from brancher.utilities import coerce_to_dtype
from brancher.utilities import broadcast_and_reshape_parent_value
from brancher.utilities import split_dict
from brancher.utilities import reformat_sampler_input
from brancher.utilities import tile_parameter
from brancher.utilities import get_model_mapping
from brancher.utilities import reassign_samples
from brancher.utilities import is_discrete, is_tensor, contains_tensors
from brancher.utilities import concatenate_samples
from brancher.utilities import get_number_samples_and_datapoints
from brancher.utilities import map_iterable
from brancher.utilities import sum_from_dim

from brancher.pandas_interface import reformat_sample_to_pandas
from brancher.pandas_interface import reformat_model_summary
from brancher.pandas_interface import pandas_frame2dict
from brancher.pandas_interface import pandas_frame2value

from brancher.config import device

#TODO: This requires some refactoring, it should become a superclass of Variables and probabilistic models and Partial Links should be out
class BrancherClass(ABC):
    """
    BrancherClass is the abstract superclass of all Brancher variables and models.
    """
    @abstractmethod
    def _flatten(self):
        """
        Abstract method. It returns a list of all the ancestor variables of the variable or contained in the model.

        Args: None.

        Returns:
            List.
        """
        pass

    @abstractmethod
    def _get_statistic(self, query, input_values):
        """
        Abstract method. It returns a statistical value of the distribution given a query and optional conditioning
        parameters.

        Args:
            query: Function. A function that has as input a distribution with its parameters and as output torch.Tensor.
            See distribution package for more details.

            input_values: Dictionary(brancher.Variable: torch.Tensor, numeric, np.ndarray). A dictionary having the
            brancher.Variables of the model as keys and torch.Tensor, np.array or number as values. This dictionary has
            to provide values for all variables of the model except for the deterministic variables.

        Returns:
            torch.Tensor.
        """
        pass

    def flatten(self):
        """
        Method. Returns set of all the ancestor variables of the variable or contained in the model.

        Args: None.

        Returns:
            Set.
        """
        return set(self._flatten())

    def get_variable(self, var_name):
        """
        It returns the variable in the model with the requested name.

        Args:
            var_name: String. Name of the requested variable.

        Returns:
            branchar.Variable.
        """
        flat_list = self._flatten()
        try:
            return {var.name: var for var in flat_list}[var_name]
        except ValueError:
            raise ValueError("The variable {} is not present in the model".format(var_name))

    def _get_mean(self, input_values={}):
        return self._get_statistic(query=lambda dist, parameters: dist.get_mean(**parameters), input_values=input_values)

    def _get_variance(self, input_values={}):
        return self._get_statistic(query=lambda dist, parameters: dist.get_variance(**parameters), input_values=input_values)

    #def _get_entropy(self, input_values={}):
    #    if isinstance(self, ProbabilisticModel) or self.distribution.has_analytic_entropy:
    #        entropy_array = self._get_statistic(query=lambda dist, parameters: dist.get_entropy(**parameters),
    #                                            input_values=input_values)
    #        if isinstance(self, ProbabilisticModel):
    #            return sum([sum_from_dim(var_ent, 2) for var_ent in entropy_array.values()])
    #        else:
    #            return sum_from_dim(entropy_array, 2)
    #    else:
    #        return -self.calculate_log_probability(self, input_values, include_parents=False)


class Variable(BrancherClass):
    """
    Variable is the abstract superclass of deterministic and random variables. Variables are the building blocks of
    all probabilistic models in Brancher.
    """
    @abstractmethod
    def calculate_log_probability(self, values, reevaluate, include_parents):
        """
        Abstract method. It returns the log probability of the values given the model.

        Args:
            values: Dictionary(brancher.Variable: torch.Tensor, numeric, np.ndarray). A dictionary having the
            brancher.Variables of the model as keys and torch.Tensor, np.array or number as values. This dictionary has
            to provide values for all variables of the model except for the deterministic variables.

            reevaluate: Bool. If false it returns the output of the latest call. It avoid unnecessary computations when
            multiple children variables ask for the log probability of the same paternt variable.

        Returns:
            torch.Tensor. the log probability of the input values given the model.
        """
        pass

    @abstractmethod
    def _get_sample(self, number_samples, resample, observed, input_values, differentiable):
        """
        Abstract private method. It returns samples from the joint distribution specified by the model. If an input is
        provided it only samples the variables that are not contained in the input.

        Args:
            number_samples: Int. Number of samples that need to be generated.

            resample: Bool. If true it returns its previously stored sampled. It is used when multiple children
            variables ask for a sample to the same parent. In this case the resample variable is False since the
            children should be fed with the same value of the parent.

            observed: Bool. It specifies whether the samples should be interpreted frequentistically as samples from the
            observations of as Bayesian samples from the prior model. The first batch dimension is reserved to Bayesian
            samples while the second batch dimension is reserved to observation samples. Theerefore, this boolean
            changes the shape of the resulting sampled array.

            input_values: Dictionary(brancher.Variable: torch.Tensor, numeric, np.ndarray). A dictionary having the
            brancher.Variables of the model as keys and torch.Tensor, np.array or numberic values. This dictionary has
            to provide values for all variables of the model that do not need to be sampled. Using an input allows to
            use a probabilistic model as a random function.

        Returns:
            Dictionary(brancher.Variable: torch.Tensor). A dictionary of samples from all the variables of the model
        """
        pass

    def _get_entropy(self, input_values={}):
        """
        Method. It returns the entropy of the variable optionally conditioned on input values.

        Args:
            input_values: Dictionary(brancher.Variable: torch.Tensor, numeric, np.ndarray). A dictionary having the
            brancher.Variables of the model as keys and torch.Tensor, np.array or number as values. This dictionary has
            to provide values for all variables of the model except for the deterministic variables.

        Returns:
            torch.Tensor or np.array.
        """
        if self.distribution.has_analytic_entropy:
            entropy_array = self._get_statistic(query=lambda dist, parameters: dist.get_entropy(**parameters),
                                                input_values=input_values)
            return sum_from_dim(entropy_array, 2)
        else:
            return -self.calculate_log_probability(input_values, include_parents=False)

    def get_sample(self, number_samples, input_values={}):
        """
        Method. It returns a user specified number of samples optionally conditioned on input values. The function
        samples its parent variables recursively unless those variables were already samples in an earlier iteration.

        Args:
            number_samples: Int. Number of samples that need to be generated.

            input_values: pandas.Dataframe or Dictionary(brancher.Variable: torch.Tensor, numeric, np.ndarray). A
            dictionary having the brancher.Variables of the model as keys and torch.Tensor, np.array or numberic values.
            This dictionary provides the values of variables to condition on.

        Returns:
            pandas.Dataframe. A pandas dataframe with a row for each sample and column for each variable in the model.
        """
        reformatted_input_values = reformat_sampler_input(pandas_frame2dict(input_values),
                                                          number_samples=number_samples)
        raw_sample = {self: self._get_sample(number_samples, resample=False,
                                             observed=self.is_observed,
                                             differentiable=False,
                                             input_values=reformatted_input_values)[self]}
        sample = reformat_sample_to_pandas(raw_sample)
        self.reset()
        return sample

    def get_mean(self, input_values={}):
        """
        Method. It returns the mean value of the variable optionally conditioned on input values of variables. The
        function iteratively gets the mean of each parent

        Args:
            input_values: pandas.Dataframe or Dictionary(brancher.Variable: torch.Tensor, numeric, np.ndarray). A
            dictionary having the brancher.Variables of the model as keys and torch.Tensor, np.array or numberic values.
            This dictionary provides the values of variables to condition on.

        Returns:
            pandas.Dataframe. A pandas dataframe with the mean or each variable
        """
        reformatted_input_values = reformat_sampler_input(pandas_frame2dict(input_values),
                                                          number_samples=1)
        raw_mean = {self: self._get_mean(reformatted_input_values)}
        mean = reformat_sample_to_pandas(raw_mean)
        return mean

    def get_variance(self, input_values={}):
        """
        Method. It returns the variance value of the variable optionally conditioned on input values of variables. The
        function iteratively gets the variance of each parent

        Args:
            input_values: pandas.Dataframe or Dictionary(brancher.Variable: torch.Tensor, numeric, np.ndarray). A
            dictionary having the brancher.Variables of the model as keys and torch.Tensor, np.array or numberic values.
            This dictionary provides the values of variables to condition on.

        Returns:
            pandas.Dataframe. A pandas dataframe with the variance of each variable
        """
        reformatted_input_values = reformat_sampler_input(pandas_frame2dict(input_values),
                                                          number_samples=1)
        raw_variance = {self: self._get_variance(reformatted_input_values)}
        variance = reformat_sample_to_pandas(raw_variance)
        return variance

    def get_entropy(self, input_values={}):
        """
        Method. It returns the entropy of the variable optionally conditioned on input values of variables. The
        function iteratively gets the entropy of each parent

        Args:
            input_values: pandas.Dataframe or Dictionary(brancher.Variable: torch.Tensor, numeric, np.ndarray). A
            dictionary having the brancher.Variables of the model as keys and torch.Tensor, np.array or numberic values.
            This dictionary provides the values of variables to condition on.

        Returns:
            pandas.Dataframe. A pandas dataframe with the variance of each variable
        """
        reformatted_input_values = reformat_sampler_input(pandas_frame2dict(input_values),
                                                          number_samples=1)
        raw_ent = {self: self._get_entropy(reformatted_input_values)}
        ent = reformat_sample_to_pandas(raw_ent)
        return ent

    @abstractmethod
    def reset(self):
        """
        Abstract method. It recursively resets the self._evaluated and self._current_value attributes of the variable
        and all downstream variables. It is used after sampling and evaluating the log probability of a model.

        Args: None.

        Returns: None.
        """
        pass

    @property
    @abstractmethod
    def is_observed(self):
        """
        Abstract property method. It returns True if the variable is observed and False otherwise.

        Args: None.

        Returns: Bool.
        """
        pass

    @property
    def is_latent(self):
        """
        Property method. It returns True if the variable is not observed and it does not have any observed ancestors.

        Args: None.

        Returns: Bool.
        """
        return not self.is_observed and all([not a.is_observed for a in self.ancestors])

    def __str__(self):
        """
        Method. Returns name of variable

        Args: None

        Returns: String
        """
        return self.name

    def _apply_operator(self, other, op):
        """
        Method. It is used for performing symbolic operations between variables. It always returns a partialLink object
        that define a mathematical operation between variables. The vars attribute of the link is the set of variables
        that are used in the operation. The fn attribute is a lambda that specify the operation as a functions between
        the values of the variables in vars and a numeric output. This is required for defining the forward pass of the
        model.

        Args:
            other: brancher.PartialLink, brancher.RandomVariable, numeric or np.ndarray.

            op: Binary operator.

        Returns: brancher.PartialLink
        """
        return var2link(self)._apply_operator(other, op)

    def __neg__(self):
        return -1*self

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
        return self.__truediv__(other) ** (-1)

    def __pow__(self, other):
        return self._apply_operator(other, operator.pow)

    def __rpow__(self, other):
        raise NotImplementedError

    def __getitem__(self, key):
        """
        Method. Returns PartialLink representing one or more of the variables of this model or a slice of this variable.

        Args:
            key. String or Iterable or numeric. The names of (parent) variables or slices of the value of this variable.

        Returns:
            brancher.PartialLink.
        """
        if isinstance(key, str):
            variable_slice = key
        elif isinstance(key, Iterable):
            variable_slice = (slice(None, None, None), *key)
        else:
            variable_slice = (slice(None, None, None), key)
        vars = {self}
        fn = lambda values: values[self][variable_slice]
        links = set()
        return PartialLink(vars=vars, fn=fn, links=links)

    def shape(self):
        """
        Method. Returns PartialLink representing the shape of the variable

        Args: None

        Returns:
            brancher.PartialLink.
        """
        vars = {self}
        fn = lambda values: values[self].shape
        links = set()
        return PartialLink(vars=vars, fn=fn, links=links)


class RootVariable(Variable):
    """
    Deterministic variables are a subclass of random variables that always return the same value. The hyper-parameters
    of a probabilistic model are usually encoded as DeterministicVariables. When the user inputs a parameter as a
    numeric value or an array, brancher created a RootVariable that store its value.

    Parameters
    ----------
    data : torch.Tensor, numeric, or np.ndarray. The value of the variable. It gets stored in the self.value
    attribute.

    name : String. The name of the variable.

    learnable : Bool. This boolean value specify if the value of the RootVariable can be updated during training.

    is_observed: Bool. This boolean indicates whether the variable is observed with the data.

    """
    def __init__(self, data, name, learnable=False, is_observed=False):
        self.name = name
        self.distribution = distributions.DeterministicDistribution()
        self._evaluated = False
        self._observed = is_observed
        self.parents = set()
        self.ancestors = set()
        self._type = "Deterministic"
        self.learnable = learnable
        self.link = None
        self._value = coerce_to_dtype(data, is_observed)
        if self.learnable:
            if not is_discrete(data):
                self._value = torch.nn.Parameter(coerce_to_dtype(data, is_observed), requires_grad=True)
                self.link = ParameterModule(self._value) # add to optimizer; opt checks links
            else:
                self.learnable = False
                warnings.warn('Currently discrete parameters are not learnable. Learnable set to False')

    def calculate_log_probability(self, values, reevaluate=True, for_gradient=False, normalized=True, include_parents=False):
        """
        Method. It returns the log probability of the values given the model. This value is always 0 since the
        probability of a deterministic variable having its value is always 1.

        Args:
            values: Dictionary(brancher.Variable: torch.Tensor, numeric, np.ndarray). A dictionary having the
            brancher.Variables of the model as keys and torch.Tensor number or np.array as values. This dictionary has
            to provide values for all variables of the model except for the deterministic variables.

            reevaluate: Bool. If false it returns the output of the latest call. It avoid unnecessary computations when
            multiple children variables ask for the log probability of the same paternt variable.

            for_gradient: Bool. Unused.

            include_parents: Bool. If True, return parents log probability + own log probability

            normalized: Bool. Unused.

        Returns:
            torch.Tensor. The log probability of the input values given the model.
        """

        return torch.Tensor(np.zeros((1, 1))).float().to(device)

    @property
    def value(self):
        """
        Method. It returns the value of the deterministic variable.

        Args: None.

        Returns:
            torch.Tensor. The value of the deterministic variable.
        """
        if self.learnable:
            return self.link()
        return self._value

    @property
    def is_observed(self):
        """
        Method. It returns whether variable is observed or not.

        Args: None.

        Returns:
            Bool.
        """
        return self._observed

    def _get_statistic(self, query, input_values):
        """
        Private method. It returns a statistical value of the distribution given a query and optional conditioning
        parameters.

        Args:
            query: Function. A function that has as input a distribution with its parameters and as output torch.Tensor.
            See distribution package for more details.

            input_values: Dictionary(brancher.Variable: torch.Tensor, numeric, np.ndarray). A dictionary having the
            brancher.Variables of the model as keys and torch.Tensor, np.array or number as values. This dictionary has
            to provide values for all variables of the model except for the deterministic variables.

        Returns:
            torch.Tensor.
        """
        parameters_dict = {"value": self.value}
        statistic = query(self.distribution, parameters_dict)
        return statistic

    def _get_sample(self, number_samples, resample=False, observed=None, input_values={}, differentiable=True):
        """                                                                                                             
        Private method. It returns samples from the joint distribution specified by the model. If an input is
        provided it only samples the variables that are not contained in the input.                                     

        Args:                                                                                                           
            number_samples: Int. Number of samples that need to be generated.                                           

            resample: Bool. If true it returns its previously stored sampled. It is used when multiple children         
            variables ask for a sample to the same parent. In this case the resample variable is False since the        
            children should be fed with the same value of the parent.                                                   

            observed: Bool. It specifies whether the samples should be interpreted frequentistically as samples from the
            observations of as Bayesian samples from the prior model. The first batch dimension is reserved to Bayesian 
            samples while the second batch dimension is reserved to observation samples. Therefore, this boolean
            changes the shape of the resulting sampled array.                                                           

            input_values: Dictionary(brancher.Variable: torch.Tensor, numeric, np.ndarray). A dictionary having the
            brancher.Variables of the model as keys and torch.Tensor, np.array or numberic values. This dictionary has  
            to provide values for all variables of the model that do not need to be sampled. Using an input allows to   
            use a probabilistic model as a random function.

            differentiable: Bool. Unused.

        Returns:                                                                                                        
            Dictionary(brancher.Variable: torch.Tensor). A dictionary of samples from all the variables of the model    
        """
        if self in input_values:
            value = input_values[self]
        else:
            value = self.value
        if not is_discrete(value):
            return {self: tile_parameter(value, number_samples=number_samples)}
        else:
            return {self: value}

    def reset(self, recursive=False):
        """
        Method. It recursively resets the self._evaluated and self._current_value attributes of the variable
        and all downstream variables. It is used after sampling and evaluating the log probability of a model. Root
        variables are not reset and do not have children to recursivly reset.

        Args:
            recursive: Bool. If true, children are also reset.

        Returns: None.
        """
        pass

    def _flatten(self):
        """
        Private method. Returns empty list of variables because root variables have no parent variables.
        """
        return []


class RandomVariable(Variable):
    """
    Random variables are the main building blocks of probabilistic models.

    Parameters
    ----------
    distribution : brancher.Distribution
        The probability distribution of the random variable.
    name : str
        The name of the random variable.
    parents : tuple of brancher variables
        A tuple of brancher.Variables that are parents of the random variable.
    link : callable
        A function Dictionary(brancher.variable: torch.Tensor) -> Dictionary(str: torch.Tensor) that maps the values of
        all the parents to a dictionary of parameters of the probability distribution. It can also contain learnable
        layers and parameters.
    """
    def __init__(self, distribution, name, parents, link):
        self.name = name
        self.distribution = distribution
        self.link = link
        self.parents = parents
        self.ancestors = None
        self._type = "Random"
        self.samples = None

        self._evaluated = False
        self._observed = False # RandomVariable: observed value + link
        self._observed_value = None # need this?
        self.dataset = None
        self.has_random_dataset = False
        self.has_observed_value = False

    @property
    def value(self):
        """
        Method. It returns the value of the random variable if observed.

        Args: None.

        Returns:
            torch.Tensor. The observed value of the random variable.
        """
        if self._observed:
            return self._observed_value
        else:
            raise AttributeError('RandomVariable has to be observed to receive value.')

    @property
    def is_observed(self):
        """
        Method. It returns whether variable is observed or not.

        Args: None.

        Returns:
            Bool.
        """
        return self._observed

    def _apply_link(self, parents_values):
        """
        Private method. Applies link function to parents values while keeping the original shape of the input.

        Args:
            parents_values: Dictionary(brancher.Variable: torch.Tensor, numeric, np.ndarray).

        Returns:
            Dictionary(brancher.Variable: torch.Tensor, numeric, or np.ndarray) with link applied to the values.
        """
        number_samples, number_datapoints = get_number_samples_and_datapoints(parents_values)
        cont_values, discrete_values = split_dict(parents_values,
                                                  condition=lambda key, val: not is_discrete(val) or contains_tensors(val))
        reshaped_dict = discrete_values
        if cont_values:
            reshaped_dict.update(map_iterable(lambda x: broadcast_and_reshape_parent_value(x, number_samples, number_datapoints),
                                              cont_values, recursive=True))
        reshaped_output = self.link(reshaped_dict)
        cast_to_new_shape = lambda tensor: tensor.view(size=(number_samples, number_datapoints) + tensor.shape[1:])
        output = {key: cast_to_new_shape(val)
                  if is_tensor(val) else map_iterable(cast_to_new_shape, val) if contains_tensors(val) else val
                  for key, val in reshaped_output.items()}
        return output

    def _get_parameters_from_input_values(self, input_values):
        """
        Private method. It returns a dict of values for all parents of this variable with this variable's link applied
        to those values.

        Args:
            input_values: Dictionary(brancher.Variable: torch.Tensor, numeric, np.ndarray).

        Returns:
            Dictionary(str: torch.Tensor). Result of this variable's link function on all parent values.
        """
        if input_values:
            number_samples, _ = get_number_samples_and_datapoints(input_values)
        else:
            number_samples = 1
        deterministic_parents_values = {parent: parent._get_sample(number_samples, input_values=input_values)[parent] for parent in self.parents
                                        if isinstance(parent, RootVariable) or parent._type == "Deterministic node"}
        parents_input_values = {parent: parent_input for parent, parent_input in input_values.items() if parent in self.parents}
        parents_values = {**parents_input_values, **deterministic_parents_values}
        parameters_dict = self._apply_link(parents_values)
        return parameters_dict

    def _get_its_own_value_from_input(self, input_values, reevaluate):
        """
        Private method. Helper function for getting value of this variable from the input dict if it is there or else
        either sample it when it is a "Deterministic node" type or return its value

        Args:
            input_values: Dictionary(brancher.Variable: torch.Tensor, numeric, np.ndarray).

        Returns:
            torch.Tensor, numeric, or np.ndarray. The value of this variable.
        """
        if self in input_values:
            value = input_values[self]
        elif self._type == "Deterministic node":
            value = self._get_sample(1, input_values=input_values)[self]
        else:
            value = self.value
        return value

    def calculate_log_probability(self, input_values, reevaluate=True, for_gradient=False,
                                  include_parents=True, normalized=True):
        """
        Method. It returns the log probability of the values given the model. This value is always 0 since the probability
        of a deterministic variable having its value is always 1.

        Args:
            input_values: Dictionary(brancher.Variable: torch.Tensor, numeric, np.ndarray or dict). A dictionary having the
            brancher.Variables of the model as keys and torch.Tensor number or np.array as values. This dictionary has
            to provide values for all variables of the model except for the deterministic variables.

            reevaluate: Bool. If false it returns the output of the latest call. It avoid unnecessary computations when
            multiple children variables ask for the log probability of the same paternt variable.

            for_gradient: Bool. Unused.

            include_parents: Bool. If True, return parents log probability + own log probability

            normalized: Boo. Unused.

        Returns:
            torch.Tensor. The log probability of the input values given the model.
        """
        if self._evaluated and not reevaluate:
            return 0.
        value = self._get_its_own_value_from_input(input_values, reevaluate)
        self._evaluated = True
        parameters_dict = self._get_parameters_from_input_values(input_values)
        log_probability = self.distribution.calculate_log_probability(value, **parameters_dict)
        parents_log_probability = sum([parent.calculate_log_probability(input_values, reevaluate, for_gradient,
                                                                        normalized=normalized)
                                       for parent in self.parents])
        if self.is_observed:
            log_probability = log_probability.sum(dim=1, keepdim=True)
        if is_tensor(log_probability) and is_tensor(parents_log_probability):
            log_probability, parents_log_probability = partial_broadcast(log_probability, parents_log_probability)
        if include_parents:
            return log_probability + parents_log_probability
        else:
            return log_probability

    def _get_statistic(self, query, input_values):
        """
        Private method. It returns a statistical value of the distribution given a query and optional conditioning
        parameters.

        Args:
            query: Function. A function that has as input a distribution with its parameters and as output torch.Tensor.
            See distribution package for more details.

            input_values: Dictionary(brancher.Variable: torch.Tensor, numeric, np.ndarray).

        Returns:
            torch.Tensor.
        """
        parameters_dict = self._get_parameters_from_input_values(input_values)
        statistic = query(self.distribution, parameters_dict)
        return statistic

    def _get_sample(self, number_samples=1, resample=True, observed=None, input_values={}, differentiable=True):
        """
        Private method. Used internally. It returns samples from the random variable by sampling all its parents and
        itself.

        Args:
            number_samples: Int. Number of samples that need to be samples.

            resample: Bool. If false it returns the previously sampled values. It avoids that the parents of a variable
            are sampled multiple times.

            observed: Bool. It specifies if the sample should be formatted as observed data or Bayesian parameter.

            input_values: Dictionary(brancher.Variable: torch.Tensor, numeric, or np.ndarray).  dictionary of values of
            the parents. It is used for conditioning the sample on the (inputed) values of some of the parents.

            differentiable: Bool. Unused.

        Returns:
            Dictionary(brancher.Variable: torch.Tensor). A dictionary of samples from the variable and all its parents.
        """
        if observed is None:
            observed = False
        if self.samples is not None and not resample:
            return {self: self.samples[-1]}
        if not observed:
            if self in input_values:
                return {self: input_values[self]}
            else:
                var_to_sample = self
        else:
            if self.has_observed_value:
                return {self: self._observed_value}
            elif self.has_random_dataset:
                var_to_sample = self.dataset
            else:
                var_to_sample = self
        parents_samples_dict = join_dicts_list([parent._get_sample(number_samples, resample, observed,
                                                                   input_values, differentiable=differentiable)
                                                for parent in var_to_sample.parents])
        input_dict = {parent: parents_samples_dict[parent] for parent in var_to_sample.parents}
        parameters_dict = var_to_sample._apply_link(input_dict)
        sample = var_to_sample.distribution.get_sample(**parameters_dict)
        self.samples = [sample] #TODO: to fix
        output_sample = {**parents_samples_dict, self: sample}
        return output_sample

    def observe(self, data):
        """
        Method. It assigns an observed value to a RandomVariable.

        Args:
            data: torch.Tensor, numeric, or np.ndarray. Input observed data.

        Returns:
            None.
        """
        data = pandas_frame2value(data, self.name)
        if isinstance(data, RandomVariable):
            self.dataset = data
            self.has_random_dataset = True
        else:
            self._observed_value = coerce_to_dtype(data, is_observed=True)
            self.has_observed_value = True
        self._observed = True

    def unobserve(self):
        """
        Method. It marks a variable as not observed and it drops the dataset.

        Args:
            None.

        Returns:
            None.
        """
        self._observed = False
        self.has_observed_value = False
        self.has_random_dataset = False
        self._observed_value = None
        self.dataset = None

    def reset(self, recursive=True):
        """
        Method. It resets the evaluated flag of the variable and all its parents. Used after computing the
        log probability of a variable.

        Args:
            recursive: Bool. If True, all ancestors are also reset.

        Returns:
            None.
        """
        self.samples = None
        self._evaluated = False
        if recursive:
            for var in self.ancestors:
                var.samples = None
                var._evaluated = False

    def _flatten(self):
        """
        Method. It returns a sorted list of ancestors of this variable.

        Args:
            None.

        Returns:
            List.
        """
        variables = list(self.ancestors) + [self]
        return sorted(variables, key=lambda v: v.name)


class ProbabilisticModel(BrancherClass):
    """
    ProbabilisticModels are collections of Brancher variables.

    Parameters
    ----------
    variables: List(brancher.Variable). A list of random and deterministic variables.
    """
    def __init__(self, variables, is_fully_observed=False):
        self.posterior_model = None
        self.posterior_sampler = None
        self.observed_submodel = None
        self.latent_submodel = None
        self.is_transformed = False
        self.diagnostics = {}
        self._fully_observed = is_fully_observed
        self._initialize_model(variables)

    def _initialize_model(self, variables):
        """
        Method. It initializes the probabilistic model.

        Args:
            variables: List(brancher.Variable). A list of random and deterministic variables.

        Returns:
            None.
        """
        self._input_variables = self._validate_variables(variables)
        self._set_summary()
        self.variables = self.flatten()
        if not all([var.is_latent for var in self._input_variables]):
            self.update_latent_submodel()
        else: #This is required to avoid an infinite recursion
            self.latent_submodel = self
        if not all([var.is_observed for var in self._input_variables]):
            self.update_observed_submodel()
        else: #This is required to avoid an infinite recursion
            self.observed_submodel = self

    def add_variables(self, new_variables):
        """
        Method. It adds new variables to an existing model.

        Args:
            new_variables: brancher.Variable, List(brancher.Variable), Set(brancher.Variable) or brancher.ProbabilisticModel

        Returns:
            None.
        """
        if isinstance(new_variables, ProbabilisticModel):
            new_variables = new_variables.variables
        if isinstance(new_variables, (list, set)):
            new_input_variables = list(self.variables) + list(new_variables)
        elif isinstance(new_variables, Variable):
            new_input_variables = list(self.variables).append(Variable)
        else:
            raise ValueError("The input of the add_variable method should be a Variable, a set/list of variables or a ProbabilisticModel")
        self._initialize_model(new_input_variables)

    def __str__(self):
        """
        Method.

        Args:
            None.

        Returns:
            String.
        """
        return self.model_summary.__str__()

    @staticmethod
    def _validate_variables(variables):
        """
        Static private method. Checks if all variables are either RootVariables, RandomVariables or ProbabilisticModels.
        It also check if all variable names are unique.

        Args:
            variables: List(brancher.Variable)

        Returns:
            List(brancher.Variable)
        """
        for var in variables:
            if not isinstance(var, (RootVariable, RandomVariable, ProbabilisticModel)):
                raise ValueError("Invalid input type: {}".format(type(var)))
        names = [var.name for var in variables]
        if len(set(names)) != len(names):
            raise ValueError("The list of variables contains at least two variables with the same name")
        return variables

    def _set_summary(self):
        """
        Private method. Sets the summary of this probabilistic model as a pandas dataframe with the Distribution name
        and parent variables for each variable of the Probabilistic Model and whether that variable is observed.

        Args: None.

        Returns:
            None.
        """
        feature_list = ["Distribution", "Parents", "Observed"]
        var_list = self.flatten()
        var_names = [var.name for var in var_list]
        summary_data = [[var._type, var.parents, var.is_observed]
                         for var in var_list]
        self._model_summary = reformat_model_summary(summary_data, var_names, feature_list)

    @property
    def model_summary(self):
        """
        Method. Updates and returns model summary.

        Args: None.

        Returns:
            pandas.Dataframe. Pandas dataframe containing for each variable in this model the distribution name, parents
            and if it is observed.
        """
        self._set_summary()
        return self._model_summary

    @property
    def is_observed(self):
        """
        Method. It returns whether the model is fully observed or not.

        Args: None.

        Returns:
            Bool.
        """
        if self._fully_observed:
            return True
        else:
            return any([var.is_observed for var in self._flatten()])

    def observe(self, data):
        """
        Method. It assigns an observed value to a ProbabilisticModel.

        Args:
            data: torch.Tensor, numeric, or np.ndarray. Input observed data.

        Returns:
            None.
        """
        if isinstance(data, pd.DataFrame):
            data = {var_name: pandas_frame2value(data, index=var_name) for var_name in data}
        if isinstance(data, dict):
            if all([isinstance(k, Variable) for k in data.keys()]):
                data_dict = data
            if all([isinstance(k, str) for k in data.keys()]):
                data_dict = {self.get_variable(name): value for name, value in data.items()}
        else:
            raise ValueError("The input data should be either a dictionary of values or a pandas dataframe")
        for var in data_dict:
            if isinstance(var, RandomVariable):
                var.observe(data_dict[var])
        self.update_observed_submodel()
        self.update_latent_submodel()

    def update_observed_submodel(self):
        """
        Method. Extract the sub-model of observed variables.

        Args: None.

        Returns:
            None.
        """
        flattened_model = self._flatten()
        observed_variables = [var for var in flattened_model if var.is_observed]
        self.observed_submodel = ProbabilisticModel(observed_variables, is_fully_observed=True)

    def update_latent_submodel(self):
        flattened_model = self._flatten()
        latent_variables = [var for var in flattened_model
                            if var.is_latent]
        self.latent_submodel = ProbabilisticModel(latent_variables)

    def set_posterior_model(self, model, sampler=None): #TODO: Clean up code duplication
        """
        Method. Sets the posterior model for this model. This model will be used by some functions to sample the
        variable parameters before sampling those variables. The optional sampler is not used yet.

        Args:
            model: brancher.ProbabilisticModel. The probabilistic model that will be set for this model.

            sampler: brancher.ProbabilisticModel, brancher.Variable, or
                Iterable(brancher.Variable or brancher.ProbabilisticModel). Unused.

        Returns: None
        """
        self.posterior_model = PosteriorModel(posterior_model=model, joint_model=self)
        if sampler:
            if isinstance(sampler, ProbabilisticModel):
                self.posterior_sampler = PosteriorModel(sampler, joint_model=self)
            elif isinstance(sampler, Variable):
                self.posterior_sampler = PosteriorModel(ProbabilisticModel([sampler]), joint_model=self)
            elif isinstance(sampler, Iterable) and all([isinstance(subsampler, (ProbabilisticModel, Variable))
                                                        for subsampler in sampler]):
                self.posterior_sampler = [PosteriorModel(ProbabilisticModel([var]), joint_model=self)
                                          if isinstance(var, Variable) else PosteriorModel(var, joint_model=self)
                                          for var in sampler]
            else:
                raise ValueError("The sampler should be either a probabilistic model, a brancher variable or an iterable of variables and/or models")

    def calculate_log_probability(self, rv_values, for_gradient=False, normalized=True):
        """
        Method. It returns the log probability of the values given the model. This is calculated by summing all log
        probabilities of each variable in the model.

        Args:
            rv_values: Dictionary(brancher.Variable: torch.Tensor, numeric, np.ndarray or dict). A dictionary having
            the brancher.Variables of the model as keys and torch.Tensor number or np.array as values. This dictionary
            has to provide values for all variables of the model except for the deterministic variables.

            reevaluate: Bool. If false it returns the output of the latest call. It avoid unnecessary computations when
            multiple children variables ask for the log probability of the same parent variable.

            for_gradient: Bool. unused

            normilized: Bool. unused

        Returns:
            torch.Tensor. The log probability of the input values given the model.

        """
        log_probability = sum([var.calculate_log_probability(rv_values, reevaluate=False,
                                                             for_gradient=for_gradient,
                                                             normalized=normalized)
                               for var in self._input_variables])
        self.reset()
        return log_probability

    def _get_statistic(self, query, input_values):
        """
        Private method. It returns a statistical value of the distributions of all variables in the model given a query
        and optional conditioning parameters.

        Args:
            query: Function. A function that has as input a distribution with its parameters and as output torch.Tensor.
            See distribution package for more details.

            input_values: Dictionary(brancher.Variable: torch.Tensor, numeric, np.ndarray).

        Returns:
            Dictionary(brancher.Variable, torch.Tensor).
        """
        return {var: var._get_statistic(query, input_values) for var in self.variables}

    def _get_sample(self, number_samples, observed=None, input_values={}, differentiable=True):
        """
        Private method. Used internally. It returns a joint sample from all variables in the model.

        Args:
            number_samples: Int. Number of samples that need to be samples.

            resample: Bool. If false it returns the previously sampled values. It avoids that the parents of a variable
            are sampled multiple times.

            observed: Bool. It specifies if the sample should be formatted as observed data or Bayesian parameter.

            input_values: Dictionary(brancher.Variable: torch.Tensor, numeric, or np.ndarray).  dictionary of values of
            the parents. It is used for conditioning the sample on the (inputed) values of some of the parents.

            differentiable: Bool. Unused.

        Returns:
            Dictionary(brancher.Variable: torch.Tensor). A dictionary of samples from the variable and all its parents.
        """
        joint_sample = join_dicts_list([var._get_sample(number_samples=number_samples, resample=False,
                                                        observed=observed, input_values=input_values,
                                                        differentiable=differentiable)
                                        for var in self._input_variables])
        joint_sample.update(input_values)
        self.reset()
        return joint_sample

    def _get_entropy(self, input_values={}, for_gradient=True):
        """
        Private method. Calculates the entropy of the model optionally conditioned on input values. If the model is not
        transformed we take the sum of the entropy for each variable in the model. Else we calculate it with the
        negative log probability.

        Args:
            input_values: pandas.Dataframe or Dictionary(brancher.Variable: torch.Tensor, numeric, np.ndarray).

            for_gradient: Bool. Unused.

        Returns:
            torch.Tensor.
        """
        if not self.is_transformed:
            entropy_array = {var: var._get_entropy(input_values) for var in self.variables}
            return sum([sum_from_dim(var_ent, 2) for var_ent in entropy_array.values()])
        else:
            return -self.calculate_log_probability(input_values, for_gradient=for_gradient)

    def get_sample(self, number_samples, input_values={}):
        """
        Method. It returns a user specified number of samples optionally conditioned on input values. The function
        samples all variables in the model.

        Args:
            number_samples: Int. Number of samples that need to be generated.

            input_values: pandas.Dataframe or Dictionary(brancher.Variable: torch.Tensor, numeric, np.ndarray). A
            dictionary having the brancher.Variables of the model as keys and torch.Tensor, np.array or numberic values.
            This dictionary provides the values of variables to condition on.

        Returns:
            pandas.Dataframe. A pandas dataframe with a row for each sample and column for each variable in the model.
        """
        reformatted_input_values = reformat_sampler_input(pandas_frame2dict(input_values),
                                                                            number_samples=number_samples)
        raw_sample = self._get_sample(number_samples, input_values=reformatted_input_values,
                                      differentiable=False)
        sample = reformat_sample_to_pandas(raw_sample)
        return sample

    def get_mean(self, input_values={}):
        """
        Method. It returns the mean value of each variable in the model optionally conditioned on input values of
        variables. The function returns the variance for each variable in the model.

        Args:
            input_values: pandas.Dataframe or Dictionary(brancher.Variable: torch.Tensor, numeric, np.ndarray). A
            dictionary having the brancher.Variables of the model as keys and torch.Tensor, np.array or numberic values.
            This dictionary provides the values of variables to condition on.

        Returns:
            pandas.Dataframe. A pandas dataframe with the mean or each variable
        """
        reformatted_input_values = reformat_sampler_input(pandas_frame2dict(input_values),
                                                          number_samples=1)
        raw_mean = self._get_mean(reformatted_input_values)
        mean = reformat_sample_to_pandas(raw_mean, number_samples=1)
        return mean

    def get_variance(self, input_values={}):
        """
        Method. It returns the variance value of each variable in the model optionally conditioned on input values of
        variables. The function returns the variance for each variable in the model.

        Args:
            input_values: pandas.Dataframe or Dictionary(brancher.Variable: torch.Tensor, numeric, np.ndarray). A
            dictionary having the brancher.Variables of the model as keys and torch.Tensor, np.array or numberic values.
            This dictionary provides the values of variables to condition on.

        Returns:
            pandas.Dataframe. A pandas dataframe with the variance of each variable
        """
        reformatted_input_values = reformat_sampler_input(pandas_frame2dict(input_values),
                                                          number_samples=1)
        raw_variance = self._get_variance(reformatted_input_values)
        variance = reformat_sample_to_pandas(raw_variance, number_samples=1)
        return variance

    def get_entropy(self, input_values={}):
        """
        Method. It returns the entropy of the model optionally conditioned on input values of variables. The
        function returns the sum of the entropy of the variables in the model.

        Args:
            input_values: pandas.Dataframe or Dictionary(brancher.Variable: torch.Tensor, numeric, np.ndarray). A
            dictionary having the brancher.Variables of the model as keys and torch.Tensor, np.array or numberic values.
            This dictionary provides the values of variables to condition on.

        Returns:
            pandas.Dataframe. A pandas dataframe with the entropy of each variable
        """
        reformatted_input_values = reformat_sampler_input(pandas_frame2dict(input_values),
                                                          number_samples=1)
        raw_ent = self._get_entropy(reformatted_input_values)
        ent = reformat_sample_to_pandas(raw_ent, number_samples=1)
        return ent

    def check_posterior_model(self):
        """
        Method. Raises an error if the posterior model is not set.
        """
        if not self.posterior_model:
            raise AttributeError("The posterior model has not been initialized.")

    def get_posterior_mean(self, query, input_values):
        return self.posterior_model.get_mean(input_values)

    def get_posterior_variance(self, query, input_values):
        return self.posterior_model.get_variance(input_values)

    def get_posterior_entropy(self, query, input_values):
        return self.posterior_model.get_entropy(input_values)

    def _get_posterior_sample(self, number_samples, input_values={}, differentiable=True):
        """
        Summary
        """
        self.check_posterior_model()
        posterior_sample = self.posterior_model._get_posterior_sample(number_samples=number_samples,
                                                                      input_values=input_values,
                                                                      differentiable=differentiable)
        sample = self._get_sample(number_samples, input_values=posterior_sample, differentiable=differentiable)
        return sample

    def get_posterior_sample(self, number_samples, input_values={}):
        reformatted_input_values = reformat_sampler_input(pandas_frame2dict(input_values),
                                                                            number_samples=number_samples)
        raw_sample = self._get_posterior_sample(number_samples, input_values=reformatted_input_values)
        sample = reformat_sample_to_pandas(raw_sample)
        return sample

    def get_p_log_probabilities_from_q_samples(self, q_samples, q_model, empirical_samples={},
                                               for_gradient=False, normalized=True):
        p_samples = reassign_samples(q_samples, source_model=q_model, target_model=self)
        p_samples.update(empirical_samples)
        p_log_prob = self.calculate_log_probability(p_samples, for_gradient=for_gradient, normalized=normalized)
        return p_log_prob

    def get_importance_weights(self, q_samples, q_model, empirical_samples={},
                               for_gradient=False, give_normalization=False):
        if not empirical_samples:
            empirical_samples = self.observed_submodel._get_sample(1, observed=True, differentiable=False)
        q_log_prob = q_model.calculate_log_probability(q_samples,
                                                       for_gradient=for_gradient, normalized=False)
        p_log_prob = self.get_p_log_probabilities_from_q_samples(q_samples=q_samples,
                                                                 q_model=q_model,
                                                                 empirical_samples=empirical_samples,
                                                                 for_gradient=for_gradient,
                                                                 normalized=False)
        log_weights = (p_log_prob - q_log_prob).detach().numpy()
        alpha = np.max(log_weights)
        norm_log_weights = log_weights - alpha
        weights = np.exp(norm_log_weights)
        norm = np.sum(weights)
        weights /= norm
        if not give_normalization:
            return weights
        else:
            return weights, np.log(norm) + alpha #TODO: work in progress

    def estimate_log_model_evidence(self, number_samples, method="ELBO", input_values={},
                                    for_gradient=False, posterior_model=(), gradient_estimator=None):
        if not posterior_model:
            self.check_posterior_model()
            posterior_model = self.posterior_model
        if method == "ELBO":
            empirical_samples = self.observed_submodel._get_sample(1, observed=True, differentiable=False) #TODO Important!!: You need to correct for subsampling
            if for_gradient:
                function = lambda samples: self.get_p_log_probabilities_from_q_samples(q_samples=samples,
                                                                                       empirical_samples=empirical_samples,
                                                                                       for_gradient=for_gradient,
                                                                                       q_model=posterior_model) + posterior_model._get_entropy(samples,
                                                                                                                                               for_gradient=for_gradient)
                estimator = gradient_estimator(function, posterior_model, empirical_samples)
                log_model_evidence = estimator(number_samples)
            else:
                posterior_samples = posterior_model._get_sample(number_samples=number_samples,
                                                                observed=False, input_values=input_values,
                                                                differentiable=False)
                joint_log_prob = self.get_p_log_probabilities_from_q_samples(q_samples=posterior_samples,
                                                                             empirical_samples=empirical_samples,
                                                                             for_gradient=for_gradient,
                                                                             q_model=posterior_model)
                posterior_entropy = posterior_model._get_entropy(posterior_samples, for_gradient=for_gradient)
                log_model_evidence = torch.mean(joint_log_prob + posterior_entropy)
            return log_model_evidence
        else:
            raise NotImplementedError("The requested estimation method is currently not implemented.")

    def reset(self):
        """
        Summary
        """
        for var in self.flatten():
            var.reset(recursive=False)

    def _flatten(self):
        variables = list(join_sets_list([var.ancestors.union({var}) for var in self._input_variables]))
        sorted_variables = sorted(variables, key=lambda v: v.name)
        if self._fully_observed:
            return [var for var in sorted_variables if var.is_observed]
        else:
            return sorted_variables

#    def copy(self):
#        pass


class PosteriorModel(ProbabilisticModel):
    """
    Summary

    Parameters
    ----------
    variables : tuple of brancher variables
        Summary
    """
    def __init__(self, posterior_model, joint_model):
        super().__init__(posterior_model.variables)
        self.posterior_model = None
        self.model_mapping = get_model_mapping(self, joint_model)
        self._is_trained = False
        self.joint_model = joint_model

    def add_variables(self, new_variables):
        super().add_variables(new_variables)
        self.model_mapping = get_model_mapping(self, self.joint_model)

    def posterior_sample2joint_sample(self, posterior_sample):
        return reassign_samples(posterior_sample, self.model_mapping)

    def _get_posterior_sample(self, number_samples, observed=None, input_values={}, differentiable=True):
        sample = self.posterior_sample2joint_sample(self._get_sample(number_samples, observed, input_values,
                                                                     differentiable=differentiable))
        sample.update(input_values)
        return sample


def var2link(var):
    """
    Function. Constructs a partialLink from variables, numbers, numpy arrays, tensors or a combination of variables and
    partialLinks.

    Args:
        var: brancher.Variables, numbers, numpy.ndarrays, torch.Tensors, or List/Tuple of brancher.Variables and
        brancher.PartialLinks.

    Retuns: brancher.PartialLink
    """
    if isinstance(var, Variable):
        vars = {var}
        fn = lambda values: values[var]
    elif isinstance(var, (numbers.Number, np.ndarray, torch.Tensor)):
        vars = set()
        fn = lambda values: var
    elif isinstance(var, (tuple, list)) and all([isinstance(v, (Variable, PartialLink)) for v in var]):
        vars = join_sets_list([{v} if isinstance(v, Variable) else v.vars for v in var])
        fn = lambda values: tuple([values[v] if isinstance(v, Variable) else v.fn(values) for v in var])
    else:
        return var
    return PartialLink(vars=vars, fn=fn, links=set(), string=str(var))


class Ensemble(BrancherClass):
    """
    Ensembles are collections of models. Each model can have a weight and models can share variables.

    Parameters
    ----------
    model_list: Iterable(brancher.ProbabilisticModel). A collection of ProbabilisticModels.
    """
    def __init__(self, model_list, weights=None):
        #TODO: assert that all variables have the same name
        self.num_models = len(model_list)
        self.model_list = model_list
        if weights is None:
            self.weights = [1./self.num_models]*self.num_models
        else:
            self.weights = np.array(weights)

    def _get_sample(self, number_samples, observed=None, input_values={}, differentiable=True):
        num_samples_list = np.random.multinomial(number_samples, self.weights)
        samples_list = [model._get_sample(n, differentiable=differentiable)
                        for n, model in zip(num_samples_list, self.model_list)]
        named_sample_list = [{var.name: value for var, value in sample.items()} for sample in samples_list]
        named_sample = concatenate_samples(named_sample_list)
        sample = {self.model_list[0].get_variable(name): value for name, value in named_sample.items()}
        return sample

    # def calculate_log_probability(self, rv_values, for_gradient=False, normalized=True):
    #     """
    #     Summary
    #     """
    #     log_probabilities = [model.calculate_log_probability(rv_values, reevaluate=False,
    #                                                          for_gradient=for_gradient,
    #                                                          normalized=normalized)
    #                          for model in self.model_list]
    #     alpha = np.max(log_probabilities)


    def _flatten(self):
        return [model.flatten() for model in self.model_list]

    def get_sample(self, number_samples, input_values={}): #TODO: code duplication here
        reformatted_input_values = reformat_sampler_input(pandas_frame2dict(input_values),
                                                                            number_samples=number_samples)
        raw_sample = self._get_sample(number_samples, observed=None, input_values=reformatted_input_values,
                                      differentiable=False)
        sample = reformat_sample_to_pandas(raw_sample)
        return sample

    def _get_statistic(self, query, input_values):
        raise NotImplemented

    def _get_mean(self, input_values):
        raise NotImplemented

    def _get_variance(self, input_values):
        raise NotImplemented


class PartialLink(BrancherClass):

    def __init__(self, vars, fn, links, string=""):
        self.vars = vars
        self.fn = fn
        self.links = links
        self.string = string

    def __str__(self):
        """
        Method.

        Args: None

        Returns: String
        """
        return self.string

    def _apply_operator(self, other, op):
        symbols = {operator.add: "+", operator.sub: "-", operator.mul: "*", operator.truediv: "/", operator.pow: "**"}
        get_symbol = lambda op: symbols[op] if op in symbols.keys() else "?"
        other = var2link(other)
        return PartialLink(vars=self.vars.union(other.vars),
                           fn=lambda values: op(self.fn(values), other.fn(values)),
                           links=self.links.union(other.links),
                           string="(" + str(self) + get_symbol(op) + str(other) + ")")

    def __neg__(self):
        return -1*self

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
        return self.__truediv__(other)**(-1)

    def __pow__(self, other):
        return self._apply_operator(other, operator.pow)

    def __rpow__(self, other):
        raise NotImplementedError

    def __getitem__(self, key):
        if isinstance(key, collections.Iterable) and all([isinstance(k, int) for k in key]):
            variable_slice = (slice(None, None, None), *key)
        elif isinstance(key, int):
            variable_slice = (slice(None, None, None), key)
        elif isinstance(key, collections.Hashable):
            variable_slice = key
        else:
            raise ValueError("The input to __getitem__ is neither numeric nor a hashabble key")

        vars = self.vars
        fn = lambda values: self.fn(values)[variable_slice] if is_tensor(self.fn(values)) \
            else self.fn(values)[key]
        links = set()
        return PartialLink(vars=vars,
                           fn=fn,
                           links=self.links)

    def shape(self):
        vars = self.vars
        fn = lambda values: self.fn(values).shape
        links = set()
        return PartialLink(vars=vars,
                           fn=fn,
                           links=self.links)

    def _flatten(self):
        return flatten_list([var._flatten() for var in self.vars]) + [self]

    def _get_statistic(self, query, input_values):
        raise NotImplemented

    def _get_mean(self, input_values):
        raise NotImplemented

    def _get_variance(self, input_values):
        raise NotImplemented
