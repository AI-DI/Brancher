import types
import torch

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
        return self.name + "(" + ", ".join([var2link(a).string
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


##
# redefined functions: overwriten from torch._C._VariableFunctions by torch.nn.functional

# for name1 in dir(torch._C._VariableFunctions):
#     for name2, fun in torch.nn.functional.__dict__.items():
#         if name1 == name2:
#             print(name1, getattr(torch.Tensor, name), name2, fun)


# adaptive_avg_pool1d <function Tensor.potrf at 0x0000000007FE1378> adaptive_avg_pool1d <built-in method adaptive_avg_pool1d of type object at 0x000007FEB7E49F80>
# adaptive_max_pool1d <function Tensor.potrf at 0x0000000007FE1378> adaptive_max_pool1d <function boolean_dispatch.<locals>.fn at 0x00000000095E7BF8>
# alpha_dropout <function Tensor.potrf at 0x0000000007FE1378> alpha_dropout <function alpha_dropout at 0x00000000095E9598>
# avg_pool1d <function Tensor.potrf at 0x0000000007FE1378> avg_pool1d <built-in method avg_pool1d of type object at 0x000007FEB7E49F80>
# batch_norm <function Tensor.potrf at 0x0000000007FE1378> batch_norm <function batch_norm at 0x00000000095ED840>
# bilinear <function Tensor.potrf at 0x0000000007FE1378> bilinear <function bilinear at 0x00000000095ED488>
# binary_cross_entropy_with_logits <function Tensor.potrf at 0x0000000007FE1378> binary_cross_entropy_with_logits <function binary_cross_entropy_with_logits at 0x00000000095F0488>
# broadcast_tensors <function Tensor.potrf at 0x0000000007FE1378> broadcast_tensors <function broadcast_tensors at 0x0000000007FE2D90>
# btrifact <function Tensor.potrf at 0x0000000007FE1378> btrifact <function btrifact at 0x000000000842DBF8>
# celu <function Tensor.potrf at 0x0000000007FE1378> celu <function celu at 0x00000000095EC1E0>
# celu_ <function Tensor.potrf at 0x0000000007FE1378> celu_ <built-in method celu_ of type object at 0x000007FEB7E49F80>
# chain_matmul <function Tensor.potrf at 0x0000000007FE1378> chain_matmul <function chain_matmul at 0x0000000009740598>
# conv1d <function Tensor.potrf at 0x0000000007FE1378> conv1d <built-in method conv1d of type object at 0x000007FEB7E49F80>
# conv2d <function Tensor.potrf at 0x0000000007FE1378> conv2d <built-in method conv2d of type object at 0x000007FEB7E49F80>
# conv3d <function Tensor.potrf at 0x0000000007FE1378> conv3d <built-in method conv3d of type object at 0x000007FEB7E49F80>
# conv_tbc <function Tensor.potrf at 0x0000000007FE1378> conv_tbc <built-in method conv_tbc of type object at 0x000007FEB7E49F80>
# conv_transpose1d <function Tensor.potrf at 0x0000000007FE1378> conv_transpose1d <built-in method conv_transpose1d of type object at 0x000007FEB7E49F80>
# conv_transpose2d <function Tensor.potrf at 0x0000000007FE1378> conv_transpose2d <built-in method conv_transpose2d of type object at 0x000007FEB7E49F80>
# conv_transpose3d <function Tensor.potrf at 0x0000000007FE1378> conv_transpose3d <built-in method conv_transpose3d of type object at 0x000007FEB7E49F80>
# cosine_embedding_loss <function Tensor.potrf at 0x0000000007FE1378> cosine_embedding_loss <function cosine_embedding_loss at 0x00000000095F2048>
# cosine_similarity <function Tensor.potrf at 0x0000000007FE1378> cosine_similarity <built-in method cosine_similarity of type object at 0x000007FEB7E49F80>
# ctc_loss <function Tensor.potrf at 0x0000000007FE1378> ctc_loss <function ctc_loss at 0x00000000095EDD90>
# dropout <function Tensor.potrf at 0x0000000007FE1378> dropout <function dropout at 0x00000000095E9488>
# einsum <function Tensor.potrf at 0x0000000007FE1378> einsum <function einsum at 0x0000000009727EA0>
# embedding <function Tensor.potrf at 0x0000000007FE1378> embedding <function embedding at 0x00000000095ED620>
# embedding_bag <function Tensor.potrf at 0x0000000007FE1378> embedding_bag <function embedding_bag at 0x00000000095ED730>
# feature_alpha_dropout <function Tensor.potrf at 0x0000000007FE1378> feature_alpha_dropout <function feature_alpha_dropout at 0x00000000095E98C8>
# group_norm <function Tensor.potrf at 0x0000000007FE1378> group_norm <function group_norm at 0x00000000095EDB70>
# hardshrink <function Tensor.potrf at 0x0000000007FE1378> hardshrink <function hardshrink at 0x00000000095EC620>
# hinge_embedding_loss <function Tensor.potrf at 0x0000000007FE1378> hinge_embedding_loss <function hinge_embedding_loss at 0x00000000095F0B70>
# instance_norm <function Tensor.potrf at 0x0000000007FE1378> instance_norm <function instance_norm at 0x00000000095ED950>
# kl_div <function Tensor.potrf at 0x0000000007FE1378> kl_div <function kl_div at 0x00000000095F0158>
# layer_norm <function Tensor.potrf at 0x0000000007FE1378> layer_norm <function layer_norm at 0x00000000095EDA60>
# log_softmax <function Tensor.potrf at 0x0000000007FE1378> log_softmax <function log_softmax at 0x00000000095ED048>
# margin_ranking_loss <function Tensor.potrf at 0x0000000007FE1378> margin_ranking_loss <function margin_ranking_loss at 0x00000000095F0A60>
# max_pool1d_with_indices <function Tensor.potrf at 0x0000000007FE1378> max_pool1d_with_indices <function max_pool1d_with_indices at 0x00000000095E6AE8>
# meshgrid <function Tensor.potrf at 0x0000000007FE1378> meshgrid <function meshgrid at 0x00000000097400D0>
# mul <function Tensor.potrf at 0x0000000007FE1378> mul <built-in function mul>
# norm <function Tensor.potrf at 0x0000000007FE1378> norm <function norm at 0x0000000009740510>
# pairwise_distance <function Tensor.potrf at 0x0000000007FE1378> pairwise_distance <function pairwise_distance at 0x00000000095F27B8>
# pdist <function Tensor.potrf at 0x0000000007FE1378> pdist <built-in method pdist of type object at 0x000007FEB7E49F80>
# pixel_shuffle <function Tensor.potrf at 0x0000000007FE1378> pixel_shuffle <built-in method pixel_shuffle of type object at 0x000007FEB7E49F80>
# prelu <function Tensor.potrf at 0x0000000007FE1378> prelu <function prelu at 0x00000000095EC400>
# relu <function Tensor.potrf at 0x0000000007FE1378> relu <function relu at 0x00000000095E9AE8>
# relu_ <function Tensor.potrf at 0x0000000007FE1378> relu_ <built-in method relu_ of type object at 0x000007FEB7E49F80>
# rrelu <function Tensor.potrf at 0x0000000007FE1378> rrelu <function rrelu at 0x00000000095EC510>
# rrelu_ <function Tensor.potrf at 0x0000000007FE1378> rrelu_ <built-in method rrelu_ of type object at 0x000007FEB7E49F80>
# selu <function Tensor.potrf at 0x0000000007FE1378> selu <function selu at 0x00000000095EC0D0>
# selu_ <function Tensor.potrf at 0x0000000007FE1378> selu_ <built-in method selu_ of type object at 0x000007FEB7E49F80>
# sigmoid <function Tensor.potrf at 0x0000000007FE1378> sigmoid <function sigmoid at 0x00000000095ED268>
# softmax <function Tensor.potrf at 0x0000000007FE1378> softmax <function softmax at 0x00000000095ECB70>
# split <function Tensor.potrf at 0x0000000007FE1378> split <function split at 0x000000000842DAE8>
# stft <function Tensor.potrf at 0x0000000007FE1378> stft <function stft at 0x0000000009740158>
# tanh <function Tensor.potrf at 0x0000000007FE1378> tanh <function tanh at 0x00000000095ED158>
# tensordot <function Tensor.potrf at 0x0000000007FE1378> tensordot <function tensordot at 0x0000000009740400>
# threshold <function Tensor.potrf at 0x0000000007FE1378> threshold <function threshold at 0x00000000095E99D8>
# threshold_ <function Tensor.potrf at 0x0000000007FE1378> threshold_ <built-in method threshold_ of type object at 0x000007FEB7E49F80>
# triplet_margin_loss <function Tensor.potrf at 0x0000000007FE1378> triplet_margin_loss <function triplet_margin_loss at 0x00000000095F28C8>
