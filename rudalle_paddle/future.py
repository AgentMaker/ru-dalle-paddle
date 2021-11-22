# -*- coding: utf-8 -*-
# add future ops for paddle
import warnings

import paddle
import paddle.nn.functional as F

from paddle import Tensor


set_device = paddle.set_device
paddle.set_device = lambda device: set_device(device.replace('cuda', 'gpu'))


def to(self, place, dtype=None):
    if isinstance(place, str):
        if place == 'cpu':
            place = paddle.CPUPlace()
        elif place == 'cuda':
            place = paddle.CUDAPlace(0)
        elif 'cuda:' in place:
            place = paddle.CUDAPlace(int(place.split(':')[1]))
    out = self
    if isinstance(dtype, str):
        dtype = getattr(paddle, dtype)
    if dtype is not None and self.dtype != dtype:
        out = self.astype(dtype)
    if self.place._equals(place):
        return out
    out = paddle.to_tensor(out, place=place, stop_gradient=self.stop_gradient)
    if self.grad is not None:
        grad = self.grad.to(place, dtype)
        out._set_grad_ivar(grad)
    return out


paddle.Tensor.to = to
paddle.Tensor.cpu = lambda self: to(self, 'cpu')
paddle.Tensor.cuda = lambda self: to(self, 'cuda')


_layer_to = paddle.nn.Layer.to


def layer_to(self, device=None, dtype=None, blocking=None):
    if isinstance(device, str):
        if device == 'cpu':
            place = paddle.CPUPlace()
        elif device == 'cuda':
            place = paddle.CUDAPlace(0)
        elif 'cuda:' in device:
            place = paddle.CUDAPlace(int(device.split(':')[1]))
    if isinstance(dtype, paddle.dtype):
        dtype = str(dtype).split('.')[-1]
    if self.parameters()[0].place._equals(place):
        device = None
    if self.parameters()[0].dtype == dtype:
        dtype = None
    if device is None and dtype is None:
        return self
    _layer_to(self, device, dtype, blocking)
    return self


paddle.nn.Layer.to = layer_to


def swapaxes(self, a, b):
    dims = list(range(self.ndim))
    dims[a], dims[b] = dims[b], dims[a]

    return self.transpose(dims)


paddle.Tensor.swapaxes = swapaxes


def masked_fill(self, mask, value):
    y = paddle.full(self.shape, value, self.dtype)
    return paddle.where(mask, y, self)


paddle.Tensor.masked_fill = masked_fill


def pad_sequence(sequences, batch_first=False, padding_value=0):
    max_t = max([seq.shape[0] for seq in sequences])
    _sequences = []
    for seq in sequences:
        if max_t > seq.shape[0]:
            pad_num = max_t - seq.shape[0]
            pads = paddle.to_tensor([padding_value] * pad_num, dtype=seq.dtype, place=seq.place)
            pads = pads.reshape([-1, *list(range(len(seq.shape[1:])))])
            pads = pads.expand([1, *seq.shape[1:]])
            seq = paddle.stack([seq, pads], 0 if batch_first else 1)
        else:
            seq = seq.unsqueeze(0 if batch_first else 1)
        _sequences.append(seq)

    return paddle.concat(_sequences, 0 if batch_first else 1)


paddle.pad_sequence = pad_sequence


def exponential_(self):
    eps = 1e-10
    U = paddle.rand(self.shape)
    out = -paddle.log(U + eps) + eps
    self[:] = out
    return self


paddle.Tensor.exponential_ = exponential_


paddle.Tensor.softmax = lambda self, *args, **kwargs: F.softmax(self, *args, **kwargs)


def gumbel_softmax(logits: Tensor, tau: float = 1, hard: bool = False, eps: float = 1e-10, axis: int = -1) -> Tensor:
    r"""
    Samples from the Gumbel-Softmax distribution (`Link 1`_  `Link 2`_) and optionally discretizes.

    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd
      axis (int): A dimension along which softmax will be computed. Default: -1.

    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.
      If ``hard=True``, the returned samples will be one-hot, otherwise they will
      be probability distributions that sum to 1 across `dim`.

    .. note::
      This function is here for legacy reasons, may be removed from nn.Functional in the future.

    .. note::
      The main trick for `hard` is to do  `y_hard - y_soft.detach() + y_soft`

      It achieves two things:
      - makes the output value exactly one-hot
      (since we add then subtract y_soft value)
      - makes the gradient equal to y_soft gradient
      (since we strip all other gradients)

    Examples::
        >>> logits = paddle.randn(20, 32)
        >>> # Sample soft categorical using reparametrization trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=False)
        >>> # Sample hard categorical using "Straight-through" trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=True)

    .. _Link 1:
        https://arxiv.org/abs/1611.00712
    .. _Link 2:
        https://arxiv.org/abs/1611.01144
    """
    if eps != 1e-10:
        warnings.warn('`eps` parameter is deprecated and has no effect.')

    gumbels = (
        -paddle.empty_like(logits, dtype=logits.dtype).exponential_().log()
    )  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(axis)

    if hard:
        # Straight through.
        index = y_soft.argmax(axis, keepdim=True)
        if axis < 0:
            axis = index.ndim + axis
        y_hard = F.one_hot(index, y_soft.shape[axis]).swapaxes(axis, -1)[..., 0]
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


F.gumbel_softmax = gumbel_softmax


def scatter_by_axis(self, axis, index, src):
    shapes = index.shape
    # Create additional indices
    grid_range = [paddle.arange(s) for s in shapes]
    grids = paddle.meshgrid(*grid_range)
    grids[axis] = index
    # Create final indices
    final_index = paddle.stack(grids, -1)
    # Get scatter-added tensor
    is_bool = src.dtype == paddle.bool
    if is_bool:
        src = src.astype(paddle.int64)
    scatter = paddle.scatter_nd(final_index, src, self.shape)
    if is_bool:
        scatter = scatter > 0

    return scatter


paddle.Tensor.scatter_by_axis = scatter_by_axis

pd_max_native = paddle.Tensor.max


def pd_max(self, *args, **kwargs):
    return pd_max_native(self.to(self.place, 'float32'), *args, **kwargs).to(self.place, self.dtype)


paddle.Tensor.max = pd_max

# not support fp16 yet


def layer_norm_forward(self, input):
    return F.layer_norm(
        input.to(input.place, 'float32'),
        normalized_shape=self._normalized_shape,
        weight=self.weight.to(input.place, 'float32'),
        bias=self.bias.to(input.place, 'float32'),
        epsilon=self._epsilon).to(input.place, input.dtype)


paddle.nn.LayerNorm.forward = layer_norm_forward
