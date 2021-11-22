# -*- coding: utf-8 -*-
import paddle

from . import RearrangeMixin, ReduceMixin
from ._weighted_einsum import WeightedEinsumMixin

__author__ = 'Wu Hecong'


class Rearrange(RearrangeMixin, paddle.nn.Layer):
    def forward(self, input):
        return self._apply_recipe(input)


class Reduce(ReduceMixin, paddle.nn.Layer):
    def forward(self, input):
        return self._apply_recipe(input)


class WeightedEinsum(WeightedEinsumMixin, paddle.nn.Layer):
    def _create_parameters(self, weight_shape, weight_bound, bias_shape, bias_bound):
        self.weight = self.create_parameter(weight_shape,
                                            default_initializer=paddle.initializer.Uniform(-weight_bound, weight_bound))
        if bias_shape is not None:
            self.bias = self.create_parameter(bias_shape,
                                              default_initializer=paddle.initializer.Uniform(-bias_bound, bias_bound))
        else:
            self.bias = None

    def forward(self, input):
        result = paddle.einsum(self.einsum_pattern, input, self.weight)
        if self.bias is not None:
            result += self.bias
        return result
