# -*- coding: utf-8 -*-
import paddle
from paddle import nn


def conversion_helper(val, conversion):
    """Apply conversion to val. Recursively apply conversion if `val` is a nested tuple/list structure."""
    if not isinstance(val, (tuple, list)):
        return conversion(val)
    rtn = [conversion_helper(v, conversion) for v in val]
    if isinstance(val, tuple):
        rtn = tuple(rtn)
    return rtn


def fp32_to_fp16(val):
    """Convert fp32 `val` to fp16"""
    def half_conversion(val):
        if val.dtype == paddle.float32:
            val = val.astype(paddle.float16)
        return val
    return conversion_helper(val, half_conversion)


def fp16_to_fp32(val):
    """Convert fp16 `val` to fp32"""
    def float_conversion(val):
        if isinstance(val, paddle.Tensor) and val.dtype == paddle.float16:
            val = val.astype(paddle.float32)
        return val
    return conversion_helper(val, float_conversion)


class FP16Module(nn.Layer):
    def __init__(self, module):
        super(FP16Module, self).__init__()
        self.add_sublayer('module', module.to(module.device, dtype='float16'))

    def forward(self, *inputs, **kwargs):
        return fp16_to_fp32(self.module(*(fp32_to_fp16(inputs)), **kwargs))

    def state_dict(self, destination=None, include_sublayers=True):
        return self.module.state_dict(destination, include_sublayers)

    def set_state_dict(self, state_dict, use_structured_name=True):
        self.module.set_state_dict(state_dict, use_structured_name)

    def get_param(self, item):
        return self.module.get_param(item)

    def to(self, device, *args, **kwargs):
        self.module = self.module.to(device)
        return super().to(device, *args, **kwargs)
