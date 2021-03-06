# -*- coding: utf-8 -*-
import chainer

from . import RearrangeMixin, ReduceMixin
from ._weighted_einsum import WeightedEinsumMixin

__author__ = 'Alex Rogozhnikov'


class Rearrange(RearrangeMixin, chainer.Link):
    def __call__(self, x):
        return self._apply_recipe(x)


class Reduce(ReduceMixin, chainer.Link):
    def __call__(self, x):
        return self._apply_recipe(x)


class WeightedEinsum(WeightedEinsumMixin, chainer.Link):
    def _create_parameters(self, weight_shape, weight_bound, bias_shape, bias_bound):
        uniform = chainer.variable.initializers.Uniform
        with self.init_scope():
            self.weight = chainer.variable.Parameter(uniform(weight_bound), weight_shape)
            if bias_shape is not None:
                self.bias = chainer.variable.Parameter(uniform(bias_bound), bias_shape)
            else:
                self.bias = None

    def __call__(self, input):
        result = chainer.functions.einsum(self.einsum_pattern, input, self.weight)
        if self.bias is not None:
            result = result + self.bias
        return result
