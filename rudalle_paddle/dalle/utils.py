# -*- coding: utf-8 -*-
import paddle


def exists(val):
    return val is not None


def is_empty(t):
    return all([s == 0 for s in t.shape])


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, '{} is not divisible by {}'.format(
        numerator, denominator)


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def split_tensor_along_last_dim(tensor, num_partitions, contiguous_split_chunks=False):
    """
    Split a tensor along its last dimension.
    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    # Split.
    tensor_list = paddle.split(tensor, num_partitions, axis=last_dim)
    # Note: paddle.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk for chunk in tensor_list)
    return tensor_list


def init_method_normal(std=0.02):
    """Init method based on normal distribution.

    This is only used for embeddings. The transformer has its
    own initializer.
    """
    def init_(tensor):
        return paddle.nn.initializer.Normal(mean=0.0, std=std)(tensor)
    return init_
