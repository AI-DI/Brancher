"""
Utilities
---------
Module description
"""
import sys
from functools import reduce
from collections import abc
from collections.abc import Iterable

import numpy as np
import torch
import pandas as pd

from brancher.config import device


def is_tensor(data):
    return torch.is_tensor(data)


def contains_tensors(data):
    if isinstance(data, dict):
        return all([is_tensor(d) for d in data.values()])
    if isinstance(data, Iterable):
        return all([is_tensor(d) for d in data])
    else:
        return False


def is_discrete(data):
    return type(data) in [list, set, tuple, dict, str]


def to_tuple(obj):
    if isinstance(obj, Iterable) and not isinstance(obj, torch.Tensor):
        return tuple(obj)
    else:
        return tuple([obj])


def to_tensor(arr):
    if isinstance(arr, torch.Tensor):
        return arr
    elif isinstance(arr, np.ndarray):
        return torch.Tensor(arr)
    else:
        raise ValueError("The input should be either a torch.Tensor or a np.array")


def map_iterable(func, itr, recursive=False):
    def f(x):
        if not recursive:
            return func(x)
        else:
            return map_iterable(func, x, recursive=True)
    if is_tensor(itr) or not isinstance(itr, Iterable):
        return func(itr)
    if isinstance(itr, dict):
        return {key: f(val) for key, val in itr.items()}
    out = [*map(f, itr)]
    if isinstance(itr, list):
        return out
    elif isinstance(itr, tuple):
        return tuple(out)

def zip_dict(first_dict, second_dict):
    keys = set(first_dict.keys()).intersection(set(second_dict.keys()))
    return {k: to_tuple(first_dict[k]) + to_tuple(second_dict[k]) for k in keys}


def zip_dict_list(dict_list):
    if len(dict_list) == 0:
        return {}
    if len(dict_list) == 1:
        return dict_list[0]
    else:
        zipped_dict = zip_dict(dict_list[-1], dict_list[-2])
        new_dict_list = dict_list[:-2]
        new_dict_list.append(zipped_dict)
        return zip_dict_list(new_dict_list)


def split_dict(dic, condition):
    dict_1 = {}
    dict_2 = {}
    for key, val in dic.items():
        if condition(key, val):
            dict_1.update({key: val})
        else:
            dict_2.update({key: val})
    return dict_1, dict_2


def flatten_list(lst):
    flat_list = [item for sublist in lst for item in sublist]
    return flat_list


def flatten_set(st):
    flat_set = set([item for subset in st for item in subset])
    return flat_set


def join_dicts_list(dicts_list):
    if dicts_list:
        return reduce(lambda d1, d2: {**d1, **d2}, dicts_list)
    else:
        return {}


def join_sets_list(sets_list):
    if sets_list:
        return reduce(lambda d1, d2: d1.union(d2), sets_list)
    else:
        return set()


def sum_from_dim(tensor, dim_index):
    if is_tensor(tensor):
        data_dim = len(tensor.shape)
        for dim in reversed(range(dim_index, data_dim)):
            tensor = tensor.sum(dim=dim)
        return tensor
    else:
        return np.sum(tensor, axis=tuple(range(1, len(tensor.shape) - 1)), keepdims=False)[:, 0]

def sum_data_dimensions(var):
    return sum_from_dim(var, dim_index=2)


def partial_broadcast(*args):
    assert all([is_tensor(ar) for ar in args]), 'at least 1 object is not torch tensor'
    shapes0, shapes1 = zip(*[(x.shape[0], x.shape[1]) for x in args])
    s0, s1 = np.max(shapes0), np.max(shapes1)
    return [x.expand((s0, s1) + x.shape[2:]) for x in args]


def tile_batch_dimensions(tensor, number_samples, number_datapoints):
    return tensor.expand((number_samples, number_datapoints) + tensor.shape[2:])


def broadcast_and_squeeze(*args):
    assert all([is_tensor(ar) for ar in args]), 'at least 1 object is not torch tensor'
    if all([np.prod(val.shape[2:]) == 1 for val in args]):
        args = [val.contiguous().view(size=val.shape[:2] + tuple([1, 1])) for val in args]
    uniformed_values = uniform_shapes(*args)
    broadcasted_values = torch.broadcast_tensors(*uniformed_values)
    return broadcasted_values


def broadcast_and_squeeze_mixed(tpl, dic):
    tpl_len = len(tpl)
    dict_keys, dict_values = zip(*dic.items())
    broadcasted_values = broadcast_and_squeeze(*(tpl + dict_values))
    if tpl_len > 0:
        return broadcasted_values[:tpl_len], {k: v for k, v in zip(dict_keys, broadcasted_values[tpl_len:])}
    else:
        return {k: v for k, v in zip(dict_keys, broadcasted_values[tpl_len:])}


def get_items(itr, recursive=False):
    if is_tensor(itr) or not isinstance(itr, Iterable):
        return iter

    def f(x):
        if recursive:
            return get_items(x, recursive=True)
        else:
            return x

    if isinstance(itr, dict):
        items = f(itr.items())
        itr.items()
    else:
        return f(itr)


def reshape_parent_value(value, number_samples, number_datapoints):
    newshape = tuple([number_samples * number_datapoints]) + value.shape[2:]
    return value.contiguous().view(size=newshape)


def broadcast_and_reshape_parent_value(value, number_samples, number_datapoints):
    return reshape_parent_value(tile_batch_dimensions(value, number_samples, number_datapoints),
                                number_samples, number_datapoints)


def get_number_samples_and_datapoints(parent_values):
    n_list = []
    m_list = []
    for value in parent_values.values():
        if is_tensor(value):
            n_list.append(value.shape[0])
            m_list.append(value.shape[1])
        elif contains_tensors(value):
            if isinstance(value, dict):
                itr = value.values()
            else:
                itr = value
            for tensor in itr:
                n_list.append(tensor.shape[0])
                m_list.append(tensor.shape[1])
    if not n_list and not m_list:
        return None, None
    else:
        return max(n_list), max(m_list)


def get_diagonal(tensor):
    assert torch.is_tensor(tensor), 'object is not torch tensor'
    assert tensor.ndimension() == 4, 'ndim should be equal 4'
    dim1, dim2, dim_matrix, _ = tensor.shape
    diag_ind = list(range(dim_matrix))
    expanded_diag_ind = dim1*dim2*diag_ind
    axis12_ind = [a for a in range(dim1*dim2) for _ in range(dim_matrix)]
    reshaped_tensor = tensor.contiguous().view(size=(dim1*dim2, dim_matrix, dim_matrix))
    ind = (np.array(axis12_ind), np.array(expanded_diag_ind), np.array(expanded_diag_ind))
    subdiagonal = reshaped_tensor[ind]
    return subdiagonal.view(size=(dim1, dim2, dim_matrix))


def coerce_to_dtype(data, is_observed=False):
    """Summary"""
    def reformat_tensor(result):
        if is_observed:
            result = torch.unsqueeze(result, dim=0)
            result_shape = result.shape
            if len(result_shape) == 2:
                result = result.contiguous().view(size=result_shape + tuple([1, 1]))
            elif len(result_shape) == 3:
                result = result.contiguous().view(size=result_shape + tuple([1]))
            #if len(result_shape) == 2:
            #   result = result.contiguous().view(size=result_shape + tuple([1]))
        else:
            result = torch.unsqueeze(torch.unsqueeze(result, dim=0), dim=1)
        return result

    dtype = type(data) ##TODO: do we need any additional shape checking?
    if dtype is torch.Tensor: # to tensor
        result = data.float()
    elif dtype is np.ndarray: # to tensor
        result = torch.tensor(data).float()
    elif dtype is pd.DataFrame:
        result = torch.tensor(data.values).float()
    elif dtype in [float, int] or dtype.__base__ in [np.floating, np.signedinteger]: # to tensor
        result = torch.tensor(data * np.ones(shape=(1, 1))).float()
    elif dtype in [list, set, tuple, dict, str]: # to discrete
        return data
    else:
        raise TypeError("Invalid input dtype {} - expected float, integer, np.ndarray, or torch var.".format(dtype))

    result = reformat_tensor(result)
    return result.to(device)


def tile_parameter(tensor, number_samples):
    assert is_tensor(tensor), 'object is not torch tensor'
    value_shape = tensor.shape
    if value_shape[0] == number_samples:
        return tensor
    elif value_shape[0] == 1:
        reps = tuple([number_samples] + [1] * len(value_shape[1:]))
        return tensor.repeat(repeats=reps)
    else:
        raise ValueError("The parameter cannot be broadcasted to the required number of samples")


def reformat_sampler_input(sample_input, number_samples):
    return {var: tile_parameter(coerce_to_dtype(value, is_observed=var.is_observed), number_samples=number_samples)
            for var, value in sample_input.items()}


def uniform_shapes(*args):
    assert all([is_tensor(ar) for ar in args]), 'at least 1 object is not torch tensor'
    shapes = [ar.shape for ar in args]
    max_len = np.max([len(s) for s in shapes])
    return [torch.unsqueeze(ar, dim=len(ar.shape)) if (len(ar.shape) == max_len-1) else ar
            for ar in args]


def get_model_mapping(source_model, target_model):
    model_mapping = {}
    if isinstance(target_model, dict):
        target_variables = list(target_model.keys())
    else:
        target_variables = target_model._flatten()
    for p_var in target_variables:
        try:
            model_mapping.update({source_model.get_variable(p_var.name): p_var})
        except KeyError:
            pass
    return model_mapping


def reassign_samples(samples, model_mapping=(), source_model=(), target_model=()):
    out_sample = {}
    if model_mapping:
        pass
    elif source_model and target_model:
        model_mapping = get_model_mapping(source_model, target_model)
    else:
        raise ValueError("Either a model mapping or both source and target models have to be provided as input")
    for key, value in samples.items():
        try:
            out_sample.update({model_mapping[key]: value})
        except KeyError:
            pass
    return out_sample


def reject_samples(samples, model_statistics, truncation_rule):
    decision_variable = model_statistics(samples)
    sample_indices = [index for index, value in enumerate(decision_variable) if truncation_rule(value)]
    num_accepted_samples = len(sample_indices)
    if num_accepted_samples == 0:
        return None, 0, 0.001 #TODO: Improve
    else:
        remaining_samples = {var: value[sample_indices, :] for var, value in samples.items()}

        acceptance_probability = num_accepted_samples/float(decision_variable.shape[0])
        return remaining_samples, num_accepted_samples, acceptance_probability


def concatenate_samples(samples_list):
    ''' replaced with torch.cat'''
    if len(samples_list) == 1:
        return samples_list[0]
    else:
        #num_samples = len(samples_list)
        paired_list = zip_dict_list(samples_list)
        samples = {var: torch.cat(tensor_tuple, dim=0)#.contiguous().view(tuple([num_samples]) + tuple(tensor_tuple[0].shape[1:]))
                   for var, tensor_tuple in paired_list.items()}
        return samples


def tensor_range(tensor):
    return set(np.ndarray.tolist(tensor.detach().numpy().flatten()))


def batch_meshgrid(tensor1, tensor2):
    tensor1_shape = tensor1.shape
    tensor2_shape = tensor2.shape
    new_shape = [tensor1_shape[0], tensor1_shape[1], tensor2_shape[1]]

    assert (len(tensor1_shape) == 2 and len(tensor2_shape) == 2), "You can use batch_meshgrid only on 2D tensor (The first dimension is the batch dimension)"

    tensor1 = tensor1.unsqueeze(dim=2).expand(*new_shape)
    tensor2 = tensor2.unsqueeze(dim=1).expand(*new_shape)
    return tensor1, tensor2


def get_tensor_data(tensor):
    return tensor.cpu().detach().numpy()


def delta(x, y):
    return (x == y).float()
