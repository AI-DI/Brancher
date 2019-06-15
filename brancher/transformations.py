import copy

import numpy as np

from brancher.variables import RandomVariable, ProbabilisticModel
from brancher.utilities import concatenate_samples, reject_samples
from brancher.utilities import is_tensor


def truncate_model(model, truncation_rule, model_statistics):

    def truncated_calculate_log_probability(rv_values, for_gradient=False, normalized=True):
        unnormalized_log_probability = model.calculate_log_probability(rv_values, normalized=normalized,
                                                                       for_gradient=for_gradient)
        if not normalized:
            return unnormalized_log_probability
        else:
            if for_gradient:
                nondiff_values = {var: value.detach() if is_tensor(value) else value for var, value in rv_values.items()}
                normalization = -model.calculate_log_probability(nondiff_values,
                                                                 for_gradient=False, normalized=True).mean()
                return unnormalized_log_probability + normalization
            else:
                number_samples = list(rv_values.values())[0].shape[0]
                acceptance_ratio = get_acceptance_probability(number_samples=number_samples)
                return unnormalized_log_probability - np.log(acceptance_ratio)

    def truncated_get_sample(number_samples, max_itr=1, **kwargs):
        batch_size = number_samples
        current_number_samples = 0
        sample_list = []
        itr = 0
        while (current_number_samples < number_samples and itr < max_itr) or current_number_samples < 1:
            remaining_samples, n, p = reject_samples(model._get_sample(batch_size, **kwargs),
                                                     model_statistics=model_statistics,
                                                     truncation_rule=truncation_rule)
            if remaining_samples:
                remaining_samples = {var: value[:number_samples - current_number_samples, :]
                                     for var, value in remaining_samples.items()}
                current_number_samples += np.min([n, number_samples - current_number_samples])
                sample_list.append(remaining_samples)
            itr += 1
        return concatenate_samples(sample_list)

    def get_acceptance_probability(samples=None, number_samples=None):
        if not samples:
            samples = model._get_sample(number_samples)
        _, _, p = reject_samples(samples,
                                 model_statistics=model_statistics,
                                 truncation_rule=truncation_rule)
        return p

    truncated_model = copy.copy(model)

    if isinstance(model, ProbabilisticModel):
        truncated_model.is_transformed = True
        truncated_model._get_sample = truncated_get_sample
        truncated_model.calculate_log_probability = truncated_calculate_log_probability
        truncated_model.get_acceptance_probability = get_acceptance_probability
        for var in model.variables:
            var.distribution.has_analytic_entropy = False
            var.distribution.has_analytic_mean = False
            var.distribution.has_analytic_var = False

    elif isinstance(model, RandomVariable):
        pass #TODO: Work in progress
    else:
        raise ValueError("Only probabilistic models and random variables can be truncated")
    return truncated_model
