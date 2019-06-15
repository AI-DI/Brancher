"""
Optimizers
---------
Module description
"""
from abc import ABC, abstractmethod
from collections.abc import Iterable
import copy

import torch

from brancher.standard_variables import LinkConstructor
from brancher.modules import ParameterModule, EmptyModule
from brancher.variables import BrancherClass, Variable, ProbabilisticModel

from brancher.config import device


class ProbabilisticOptimizer(ABC):
    """
    Summary

    Parameters
    ----------
    optimizer : chainer optimizer
        Summary
    """
    def __init__(self, model, optimizer='SGD', **kwargs):
        assert isinstance(optimizer, str), 'Optimizer should be a name of available pytoch optimizers'
        self.link_set = set()
        self.module = None
        self.setup(model, optimizer, **kwargs) #TODO: add asserts for checking the params dictionary

    def _update_link_set(self, model):
        assert isinstance(model, BrancherClass) #TODO: add intuitive error message
        variable_set = model.flatten() if isinstance(model, ProbabilisticModel) else model.ancestors
        for var in variable_set:
            link = var.link if hasattr(var, 'link') else None
            if isinstance(link, (ParameterModule, LinkConstructor)):  # TODO: make sure that if user inputs nn.ModuleList, this works
                self.link_set.add(link)

    def add_variable2module(self, random_variable):
        """
        Summary
        """
        self._update_link_set(random_variable)
        for link in self.link_set:
            if isinstance(link, ParameterModule):
                self.module.append(link)
            elif isinstance(link, LinkConstructor):
                [self.module.append(l) for l in link]

    def setup(self, model, optimizer, **kwargs):
        self.module = EmptyModule()
        optimizer_class = getattr(torch.optim, optimizer)
        if isinstance(model, (Variable, ProbabilisticModel)):
            self.add_variable2module(model)
        elif isinstance(model, Iterable) and all([isinstance(submodel, (Variable, ProbabilisticModel))
                                                  for submodel in model]):
            [self.add_variable2module(submodel) for submodel in model]
        else:
            raise ValueError("Only brancher variables and iterable of variables can be added to a probabilistic optimizer")
        if list(self.module.parameters()):
            self.optimizer = optimizer_class(self.module.parameters(), **kwargs)
        else:
            self.optimizer = None
        self.module.to(device)

    def update(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()
