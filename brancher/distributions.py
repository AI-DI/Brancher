"""
Distributions
---------
Module description
"""
from abc import ABC, abstractmethod
from collections.abc import Iterable
import copy

import numpy as np
from scipy.special import binom

import torch
from torch import distributions

from brancher.utilities import broadcast_and_squeeze_mixed
from brancher.utilities import broadcast_and_reshape_parent_value
from brancher.utilities import sum_data_dimensions
from brancher.utilities import is_discrete, is_tensor
from brancher.utilities import tensor_range
from brancher.utilities import map_iterable
from brancher.utilities import get_number_samples_and_datapoints

from brancher.config import device

#TODO: We need asserts checking for the right parameters

class Distribution(ABC):
    """
    Summary
    """
    def __init__(self):
        self.torchdist = None
        self.required_parameters = {}
        self.has_differentiable_samples = None
        self.is_finite = None
        self.is_discrete = None
        self.has_analytic_entropy = None
        self.has_analytic_mean = None
        self.has_analytic_var = None
        pass

    def check_parameters(self, **parameters):
        assert all([any([param in parameters for param in parameters_tuple]) if isinstance(parameters_tuple, tuple) else parameters_tuple in parameters
                    for parameters_tuple in self.required_parameters])

    @abstractmethod
    def _preprocess_parameters_for_log_prob(self, x, **parameters):
        pass

    @abstractmethod
    def _preprocess_parameters_for_sampling(self, **parameters):
        pass

    @abstractmethod
    def _postprocess_sample(self, sample, shape):
        pass

    @abstractmethod
    def _postprocess_log_prob(self, log_prob, number_samples, number_datapoints):
        pass

    def calculate_log_probability(self, x, **parameters):
        self.check_parameters(**parameters)
        x, parameters, number_samples, number_datapoints = self._preprocess_parameters_for_log_prob(x, **parameters)
        log_prob = self._calculate_log_probability(x, **parameters)
        log_prob = self._postprocess_log_prob(log_prob, number_samples, number_datapoints)
        return sum_data_dimensions(log_prob)

    def get_sample(self, differentiable=True, **parameters):
        self.check_parameters(**parameters)
        parameters, shape = self._preprocess_parameters_for_sampling(**parameters)
        pre_sample = self._get_sample(differentiable=differentiable, **parameters)
        sample = self._postprocess_sample(pre_sample, shape)
        return sample

    def get_mean(self, **parameters):
        self.check_parameters(**parameters)
        parameters, shape = self._preprocess_parameters_for_sampling(**parameters)
        pre_mean = self._get_mean(**parameters)
        mean = self._postprocess_sample(pre_mean, shape)
        return mean

    def get_variance(self, **parameters):
        self.check_parameters(**parameters)
        parameters, shape = self._preprocess_parameters_for_sampling(**parameters)
        pre_variance = self._get_variance(**parameters)
        variance = self._postprocess_sample(pre_variance, shape)
        return variance

    def get_entropy(self, **parameters):
        self.check_parameters(**parameters)
        parameters, shape = self._preprocess_parameters_for_sampling(**parameters)
        pre_entropy = self._get_entropy(**parameters)
        entropy = self._postprocess_sample(pre_entropy, shape)
        return entropy

    def _get_statistic(self, query, **parameters):
        """
        One line description

        Parameters
        ----------

        Returns
        -------
        """
        out_stat = query(self.torchdist(**parameters))
        return out_stat

    def _get_sample(self, differentiable, **parameters):
        """
        One line description

        Parameters
        ----------

        Returns
        -------
        """
        if self.has_differentiable_samples and differentiable:
            return self._get_statistic(lambda x: x.rsample(), **parameters)
        else:
            return self._get_statistic(lambda x: x.sample(), **parameters)

    def _get_mean(self, **parameters):
        """
        One line description

        Parameters
        ----------

        Returns
        -------
        """
        if self.has_analytic_mean:
            return self._get_statistic(lambda x: x.mean, **parameters)
        else:
            raise ValueError("The mean of the distribution cannot be computed analytically")

    def _get_variance(self, **parameters):
        """
        One line description

        Parameters
        ----------

        Returns
        -------
        """
        if self.has_analytic_var:
            return self._get_statistic(lambda x: x.variance, **parameters)
        raise ValueError("The variance of the distribution cannot be computed analytically")

    def _get_entropy(self, **parameters):
        """
        One line description

        Parameters
        ----------

        Returns
        -------
        """
        if self.has_analytic_entropy:
            return self._get_statistic(lambda x: x.entropy(), **parameters)
        else:
            raise ValueError("The entropy of the distribution cannot be computed analytically")

    def _calculate_log_probability(self, x, **parameters):
        """
        One line description

        Parameters
        ----------

        Returns
        -------
        """
        log_prob = self.torchdist(**parameters).log_prob(x)
        return log_prob


class ContinuousDistribution(Distribution):
    pass


class DiscreteDistribution(Distribution):
    pass


class UnivariateDistribution(Distribution):
    """
    Summary
    """

    def _preprocess_parameters_for_sampling(self, **parameters):
        parameters = broadcast_and_squeeze_mixed((), parameters)
        return parameters, None

    def _preprocess_parameters_for_log_prob(self, x, **parameters):
        tuple_x, parameters = broadcast_and_squeeze_mixed(tuple([x]), parameters)
        return tuple_x[0], parameters, None, None #TODO: add proper output here

    def _postprocess_sample(self, sample, shape=None):
        return sample

    def _postprocess_log_prob(self, log_prob, number_samples, number_datapoints):
        return log_prob


class ImplicitDistribution(Distribution):
    """
    Summary
    """

    def _preprocess_parameters_for_sampling(self, **parameters):
        return parameters, None

    def _preprocess_parameters_for_log_prob(self, x, **parameters):
        return x, parameters, None, None #TODO: add proper output here

    def _postprocess_sample(self, sample, shape=None):
        return sample

    def _calculate_log_probability(self, x, **parameters):
        return torch.tensor(np.zeros((1,1))).float().to(device) #TODO: Implement some checks here

    def _postprocess_log_prob(self, log_pro, number_samples, number_datapoints):
        return log_pro

    def _get_statistic(self, query, **parameters):
        raise NotImplemented


class VectorDistribution(Distribution):
    """
    Summary
    """
    def _preproces_vector_input(self, vector_input_dict, vector_names):
        shapes_dict = {par_name: list(par_value.shape)
                       for par_name, par_value in vector_input_dict.items()
                       if par_name in vector_names}
        reshaped_parameters = {par_name: par_value.contiguous().view(size=(shapes_dict[par_name][0], np.prod(
            shapes_dict[par_name][1:]))) if par_name in vector_names else par_value
                               for par_name, par_value in vector_input_dict.items()}
        tensor_shape = list(shapes_dict.values())[0][1:]
        return reshaped_parameters, tensor_shape

    def _preprocess_parameters_for_sampling(self, **parameters):
        number_samples, number_datapoints = get_number_samples_and_datapoints(parameters)
        parameters = map_iterable(lambda x: broadcast_and_reshape_parent_value(x, number_samples, number_datapoints), parameters)
        reshaped_parameters, tensor_shape = self._preproces_vector_input(parameters, self.vector_parameters)
        shape = tuple([number_samples, number_datapoints] + tensor_shape)
        return reshaped_parameters, shape

    def _preprocess_parameters_for_log_prob(self, x, **parameters):
        parameters_and_data = parameters
        parameters_and_data.update({"x_data": x})
        number_samples, number_datapoints = get_number_samples_and_datapoints(parameters_and_data)
        parameters_and_data = map_iterable(lambda y: broadcast_and_reshape_parent_value(y, number_samples, number_datapoints), parameters_and_data)
        vector_names = self.vector_parameters
        vector_names.add("x_data")
        reshaped_parameters_and_data, _ = self._preproces_vector_input(parameters_and_data, vector_names)
        x = reshaped_parameters_and_data.pop("x_data")
        return x, reshaped_parameters_and_data, number_samples, number_datapoints

    def _postprocess_sample(self, sample, shape):
        return sample.contiguous().view(size=shape)

    def _postprocess_log_prob(self, log_pro, number_samples, number_datapoints):
        return log_pro.contiguous().view(size=(number_samples, number_datapoints))


class CategoricalDistribution(VectorDistribution):
    """
    Summary
    """
    def __init__(self):
        super().__init__()
        self.torchdist = distributions.one_hot_categorical.OneHotCategorical
        self.required_parameters = {("probs", "logits")}
        self.optional_parameters = {}
        self.vector_parameters = {"probs", "logits"}
        self.matrix_parameters = {}
        self.scalar_parameters = {}
        self.differentiable_samples = False
        self.finite = True
        self.discrete = True
        self.analytic_entropy = True
        self.analytic_mean = True
        self.analytic_var = True

    def _calculate_log_probability(self, x, **parameters):
        """
        One line description

        Parameters
        ----------

        Returns
        -------
        """
        vector_shape = parameters["probs"].shape if "probs" in parameters else parameters["logits"].shape
        if x.shape == vector_shape and tensor_range(x) == {0, 1}:
            dist = self.torchdist
        else:
            dist = distributions.categorical.Categorical

        log_prob = dist(**parameters).log_prob(x[:, 0])
        return log_prob


class MultivariateNormalDistribution(VectorDistribution):
    """
    Summary
    """
    def __init__(self):
        super().__init__()
        self.torchdist = torch.distributions.multivariate_normal.MultivariateNormal
        self.required_parameters = {"loc", ("covariance_matrix", "precision_matrix", "scale_tril")}
        self.optional_parameters = {}
        self.vector_parameters = {"loc"}
        self.matrix_parameters = {}
        self.scalar_parameters = {}
        self.has_differentiable_samples = True
        self.is_finite = False
        self.is_discrete = False
        self.has_analytic_entropy = True
        self.has_analytic_mean = True
        self.has_analytic_var = True


class DeterministicDistribution(ImplicitDistribution):
    """
    Summary
    """
    def __init__(self):
        super().__init__()
        self.required_parameters = {"value"}
        self.has_differentiable_samples = True
        self.is_finite = True
        self.is_discrete = True
        self.has_analytic_entropy = True
        self.has_analytic_mean = True
        self.has_analytic_var = True

    def _get_sample(self, differentiable, **parameters):
        """
        One line description

        Parameters
        ----------
        Returns
        -------
        """
        return parameters["value"]

    def _get_mean(self, **parameters):
        """
        One line description

        Parameters
        ----------
        Returns
        -------
        """
        return parameters["value"]

    def _get_variance(self, **parameters):
        """
        One line description

        Parameters
        ----------
        Returns
        -------
        """
        return torch.tensor(np.zeros((1, 1, 1))).float().to(device)

    def _get_entropy(self, **parameters):
        """
        One line description

        Parameters
        ----------
        Returns
        -------
        """
        return torch.tensor(np.zeros((1, 1, 1))).float().to(device)


class EmpiricalDistribution(ImplicitDistribution): #TODO: It needs to be reworked.
    """
    Summary
    """
    def __init__(self, batch_size, is_observed):
        super().__init__()
        self.required_parameters = {"dataset"}
        self.optional_parameters = {"indices", "weights"}
        self.batch_size = batch_size
        self.is_observed = is_observed
        self.has_differentiable_samples = False
        self.is_finite = True
        self.is_discrete = True
        self.has_analytic_entropy = True #TODO: this can be implemented
        self.has_analytic_mean = False
        self.has_analytic_var = False

    def _get_sample(self, differentiable, **parameters):
        """
        One line description

        Parameters
        ----------
        Returns
        -------
        Without replacement
        """
        dataset = parameters["dataset"]
        if "indices" not in parameters:
            if "weights" in parameters:
                weights = parameters["weights"]
                p = np.array(weights).astype("float64")
                p = p/np.sum(p)
            else:
                p = None
            if is_tensor(dataset):
                if self.is_observed:
                    dataset_size = dataset.shape[1]
                else:
                    dataset_size = dataset.shape[2]
            else:
                dataset_size = len(dataset)
            if dataset_size < self.batch_size:
                raise ValueError("It is impossible to have more samples than the size of the dataset without replacement")
            if is_discrete(dataset): #
                indices = np.random.choice(range(dataset_size), size=self.batch_size, replace=False, p=p)
            else:
                number_samples = dataset.shape[0]
                indices = [np.random.choice(range(dataset_size), size=self.batch_size, replace=False, p=p)
                           for _ in range(number_samples)]
        else:
            indices = parameters["indices"]

        if is_tensor(dataset):
            if isinstance(indices, list) and isinstance(indices[0], np.ndarray):
                if self.is_observed:
                    sample = torch.cat([dataset[n, k, :].unsqueeze(dim=0) for n, k in enumerate(indices)], dim=0)
                else:
                    sample = torch.cat([dataset[n, :, k, :].unsqueeze(dim=0) for n, k in enumerate(indices)], dim=0)

            elif isinstance(indices, list) and isinstance(indices[0], (int, np.int32, np.int64)):
                if self.is_observed:
                    sample = dataset[:, indices, :]
                else:
                    sample = dataset[:, :, indices, :]
            else:
                raise IndexError("The indices of an empirical variable should be either a list of integers or a list of arrays")
        else:
            sample = list(np.array(dataset)[indices])
        return sample

    def _get_entropy(self, **parameters):
        if "weights" in parameters:
            probs = parameters["weights"]
        else:
            if is_tensor(parameters["dataset"]):
                n = int(parameters["dataset"].shape[0])
            else:
                n = len(parameters["dataset"])
            probs = torch.Tensor(np.ones((n,))).float().to(device)
        return distributions.categorical.Categorical(probs=probs).entropy()


class NormalDistribution(ContinuousDistribution, UnivariateDistribution):
    """
    Summary
    """
    def __init__(self):
        super().__init__()
        self.torchdist = distributions.normal.Normal
        self.required_parameters = {"loc", "scale"}
        self.optional_parameters = {}
        self.has_differentiable_samples = True
        self.is_finite = False
        self.is_discrete = False
        self.has_analytic_entropy = True
        self.has_analytic_mean = True
        self.has_analytic_var = True


class LogNormalDistribution(ContinuousDistribution, UnivariateDistribution):
    """
    Summary
    """
    def __init__(self):
        super().__init__()
        self.torchdist = distributions.log_normal.LogNormal
        self.required_parameters = {"loc", "scale"}
        self.optional_parameters = {}
        self.has_differentiable_samples = True
        self.is_finite = False
        self.is_discrete = False
        self.has_analytic_entropy = True
        self.has_analytic_mean = True
        self.has_analytic_var = True


class CauchyDistribution(ContinuousDistribution, UnivariateDistribution):
    """
    Summary
    """
    def __init__(self):
        super().__init__()
        self.torchdist = distributions.cauchy.Cauchy
        self.required_parameters = {"loc", "scale"}
        self.optional_parameters = {}
        self.has_differentiable_samples = True
        self.is_finite = False
        self.is_discrete = False
        self.has_analytic_entropy = True
        self.has_analytic_mean = True
        self.has_analytic_var = False


class LaplaceDistribution(ContinuousDistribution, UnivariateDistribution):
    """
    Summary
    """
    def __init__(self):
        super().__init__()
        self.torchdist = distributions.laplace.Laplace
        self.required_parameters = {"loc", "scale"}
        self.optional_parameters = {}
        self.has_differentiable_samples = True
        self.is_finite = False
        self.is_discrete = False
        self.has_analytic_entropy = True
        self.has_analytic_mean = True
        self.has_analytic_var = True


class BetaDistribution(ContinuousDistribution, UnivariateDistribution):
    """
    Summary
    """
    def __init__(self):
        super().__init__()
        self.torchdist = distributions.beta.Beta
        self.required_parameters = {"concentration1", "concentration0"}
        self.optional_parameters = {}
        self.has_differentiable_samples = True
        self.is_finite = False
        self.is_discrete = False
        self.has_analytic_entropy = True
        self.has_analytic_mean = True
        self.has_analytic_var = True


class BinomialDistribution(UnivariateDistribution, DiscreteDistribution):
    """
    Summary
    """
    def __init__(self):
        super().__init__()
        self.torchdist = distributions.binomial.Binomial
        self.required_parameters = {"total_count", ("probs", "logits")}
        self.optional_parameters = {}
        self.has_differentiable_samples = False
        self.is_finite = True
        self.is_discrete = True
        self.has_analytic_entropy = False
        self.has_analytic_mean = True
        self.has_analytic_var = True


class BernulliDistribution(UnivariateDistribution, DiscreteDistribution):
    """
    Summary
    """
    def __init__(self):
        super().__init__()
        self.torchdist = distributions.bernoulli.Bernoulli
        self.required_parameters = {("probs", "logits")}
        self.optional_parameters = {}
        self.has_differentiable_samples = False
        self.is_finite = True
        self.is_discrete = True
        self.has_analytic_entropy = True
        self.has_analytic_mean = True
        self.has_analytic_var = True



