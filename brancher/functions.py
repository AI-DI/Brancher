import types
from copy import copy

import torch
import numpy as np

from brancher.variables import var2link
from brancher.variables import Variable, PartialLink
from brancher.utilities import batch_meshgrid as torch_batch_meshgrid
from brancher.utilities import delta as torch_delta

class BrancherFunction(object):
    """
    Wrapper on backend functions (torch) for the user interface
    ----------
    """

    def __init__(self, fn, name="f_?"):
        self.fn = fn
        self.name = name
        if isinstance(fn, (torch.nn.Module, torch.nn.Sequential, torch.nn.ModuleList, torch.nn.ModuleDict,
                           torch.nn.Parameter, torch.nn.ParameterDict, torch.nn.ParameterList)): #TODO: all optimizable types in nn
            self.links = {fn}
        else:
            self.links = set()

    def _get_string(self, *args, **kwargs):
        return self.name + "(" + ", ".join([var2link(a).__str__()
                                            for n, a in enumerate(list(args) + list(kwargs.values()))]) + ")"

    def __call__(self, *args, **kwargs):
        link_args = [var2link(arg) for arg in args]
        link_kwargs = {name: var2link(arg) for name, arg in kwargs.items()}
        arg_vars = {var for link in link_args if isinstance(link, PartialLink) for var in link.vars}
        kwarg_vars = {var for _, link in link_kwargs.items() if isinstance(link, PartialLink) for var in link.vars}

        def fn(values):
            args = [x.fn(values) if isinstance(x, PartialLink) else x for x in link_args]
            kwargs = dict({(name, x.fn(values)) if isinstance(x, PartialLink) else (name, x)
                           for name, x in link_kwargs.items()})
            return self.fn(*args, **kwargs)

        return PartialLink(arg_vars.union(kwarg_vars), fn,
                           self.links, string=self._get_string(*args, **kwargs))

    @staticmethod
    def _is_var(self, arg):
        return isinstance(arg, (Variable, PartialLink))

# function set as a combination of torch.nn.functional and torch._C._VariableFunctions
# comprises most operations on tensors, including math, reshashing, broadcasting and nn functions

fn_set = {}

for name, fun in torch.nn.functional.__dict__.items():
    fn_set[name] = getattr(torch.torch.nn.functional, name)

for name in dir(torch._C._VariableFunctions):
    if name.startswith('_'):
        continue
    fn_set[name] = getattr(torch.torch._C._VariableFunctions, name)

is_backend_fn = lambda k, v: type(v) in [types.FunctionType, types.BuiltinFunctionType] and not k.startswith('_') #TODO: Work in progress
brancher_fns = {name: BrancherFunction(v, name) for name, v in fn_set.items() if is_backend_fn(name, v)}
globals().update(brancher_fns)

## Custom functions ##
batch_meshgrid = BrancherFunction(torch_batch_meshgrid)
delta = BrancherFunction(torch_delta)


## Add batch dimension to NN weights in PyTorch convolutions ##
def __batch_conv(conv_func, *args, **kwargs):
    args = list(args)
    inpt_in_args = 0
    if "input" in kwargs:
        in_data = kwargs.pop("input")
    else:
        in_data = args.pop(0)
        inpt_in_args += 1
    if "weight" in kwargs:
        weights = kwargs.pop("weight")
    else:
        weights = args.pop(1 - inpt_in_args)
    args = tuple(args)
    if "groups" in kwargs:
        raise ValueError("Group convolutions are currently not supported in Brancher.")
    in_channels = in_data.shape[1]
    out_channels = weights.shape[1]
    data_size = in_data.shape[2:]
    kernel_size = weights.shape[3:]
    batch_size = in_data.shape[0]
    in_data_t = in_data.view((1, batch_size*in_channels) + data_size)
    weights_t = weights.view((batch_size*out_channels, in_channels) + kernel_size)
    out_t = conv_func(in_data_t, weights_t, groups=batch_size, *args, **kwargs)
    out_size = out_t.shape[2:]
    out = out_t.view((batch_size, out_channels) + out_size)
    return out


def __batch_conv1d(*args, **kwargs):
    return __batch_conv(torch.nn.functional.conv1d, *args, **kwargs)


def __batch_conv2d(*args, **kwargs):
    return __batch_conv(torch.nn.functional.conv2d, *args, **kwargs)


def __batch_conv3d(*args, **kwargs):
    return __batch_conv(torch.nn.functional.conv3d, *args, **kwargs)


def __batch_conv_transpose1d(*args, **kwargs):
    return __batch_conv(torch.nn.functional.conv_transpose1d, *args, **kwargs)


def __batch_conv_transpose2d(*args, **kwargs):
    return __batch_conv(torch.nn.functional.conv_transpose2d, *args, **kwargs)


def __batch_conv_transpose3d(*args, **kwargs):
    return __batch_conv(torch.nn.functional.conv_transpose3d, *args, **kwargs)


conv1d = BrancherFunction(__batch_conv1d)
conv2d = BrancherFunction(__batch_conv2d)
conv3d = BrancherFunction(__batch_conv3d)

conv_transpose1d = BrancherFunction(__batch_conv_transpose1d)
conv_transpose2d = BrancherFunction(__batch_conv_transpose2d)
conv_transpose3d = BrancherFunction(__batch_conv_transpose3d)

## Lynear layers ##

def linear(input, weight, bias=None):
    return matmul(weight, input) + bias

## Reshape functions ##

def __reshape(input, shape):
    batch_size = input.shape[0]
    return torch.reshape(input, tuple([batch_size]) + shape)

reshape = BrancherFunction(__reshape)

## Triangular matrices ##
def _triangular_form(v):
    b_size = v.shape[0]
    N = v.shape[1]
    # TODO; assert shape
    M = int((np.sqrt(1 + 8 * N) - 1) / 2)
    tril_matrix = torch.zeros((b_size, M, M))
    tril_indices = torch.tril_indices(row=M, col=M, offset=0)
    tril_matrix[:, tril_indices[0], tril_indices[1]] = v
    return tril_matrix

triangular_form = BrancherFunction(_triangular_form)

## Matrices

def _transpose_matrix(M):
    return torch.transpose(M, 2, 1)

transpose_matrix = BrancherFunction(_transpose_matrix)

## Dot product

def _dot(v, w, reduce=True):
    b_size = v.shape[0]
    v_size = v.shape[1]
    vectors = []
    for tensor in [v, w]:
        if len(tensor.shape) == 2:
            vectors.append(torch.unsqueeze(tensor, 2))
        elif tensor.shape[2] != 1 or len(tensor.shape) > 3:
            raise ValueError("The doct product is only defined between two vectors")
        else:
            vectors.append(tensor)
    out = torch.matmul(torch.transpose(vectors[0], 2, 1), vectors[1])
    if reduce:
        return out
    else:
        return out.repeat(1, v_size, 1)

dot = BrancherFunction(_dot)