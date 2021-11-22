# -*- coding: utf-8 -*-
from abc import ABC

import paddle


class LogitsWarper(ABC):
    """Abstract base class for all logit warpers that can be applied during generation with multinomial sampling."""

    def __call__(self, input_ids, scores):
        """Paddle method for warping logits."""
        raise NotImplementedError(
            f'{self.__class__} is an abstract class. Only classes inheriting this class can be called.'
        )


class TopPLogitsWarper(LogitsWarper):
    """
    :class:`transformers.LogitsWarper` that performs top-p, i.e. restricting to top tokens summing to prob_cut_off <=
    prob_cut_off.
    Args:
        top_p (:obj:`float`):
            If set to < 1, only the most probable tokens with probabilities that add up to :obj:`top_p` or higher are
            kept for generation.
        filter_value (:obj:`float`, `optional`, defaults to :obj:`-float("Inf")`):
            All filtered values will be set to this float value.
        min_tokens_to_keep (:obj:`int`, `optional`, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(self, top_p: float, filter_value: float = -float('Inf'), min_tokens_to_keep: int = 1):
        top_p = float(top_p)
        if top_p < 0 or top_p > 1.0:
            raise ValueError(f'`top_p` has to be a float > 0 and < 1, but is {top_p}')

        self.top_p = top_p
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: paddle.Tensor, scores: paddle.Tensor) -> paddle.Tensor:
        sorted_logits = paddle.sort(scores, descending=True)
        sorted_indices = paddle.argsort(scores, descending=True)
        cumulative_probs = sorted_logits.softmax(axis=-1).cumsum(axis=-1)

        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > self.top_p
        if self.min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., : self.min_tokens_to_keep - 1] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        # unknow why package transformers use dim as `1`
        indices_to_remove = sorted_indices_to_remove.scatter_by_axis(-1, sorted_indices, sorted_indices_to_remove)
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


class TopKLogitsWarper(LogitsWarper):
    r"""
    :class:`transformers.LogitsWarper` that performs top-k, i.e. restricting to the k highest probability elements.
    Args:
        top_k (:obj:`int`):
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        filter_value (:obj:`float`, `optional`, defaults to :obj:`-float("Inf")`):
            All filtered values will be set to this float value.
        min_tokens_to_keep (:obj:`int`, `optional`, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(self, top_k: int, filter_value: float = -float('Inf'), min_tokens_to_keep: int = 1):
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f'`top_k` has to be a strictly positive integer, but is {top_k}')

        self.top_k = top_k
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: paddle.Tensor, scores: paddle.Tensor) -> paddle.Tensor:
        top_k = min(max(self.top_k, self.min_tokens_to_keep), scores.shape[-1])  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = scores < paddle.topk(scores, top_k)[0][..., -1, None]
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


def top_k_top_p_filtering(
    logits: paddle.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float('Inf'),
    min_tokens_to_keep: int = 1,
) -> paddle.Tensor:
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        top_k (:obj:`int`, `optional`, defaults to 0):
            If > 0, only keep the top k tokens with highest probability (top-k filtering)
        top_p (:obj:`float`, `optional`, defaults to 1.0):
            If < 1.0, only keep the top tokens with cumulative probability >= top_p (nucleus filtering). Nucleus
            filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        min_tokens_to_keep (:obj:`int`, `optional`, defaults to 1):
            Minimumber of tokens we keep per batch example in the output.
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        logits = TopKLogitsWarper(top_k=top_k, filter_value=filter_value, min_tokens_to_keep=min_tokens_to_keep)(
            None, logits
        )

    if 0 <= top_p <= 1.0:
        logits = TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=min_tokens_to_keep)(None, logits)

    return logits
