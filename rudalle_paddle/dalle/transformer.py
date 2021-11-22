# -*- coding: utf-8 -*-
import math

import paddle
from paddle.nn import LayerNorm

from .utils import divide, split_tensor_along_last_dim


def gelu_impl(x):
    """OpenAI's gelu implementation."""
    return 0.5 * x * (1.0 + paddle.tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x)))


def gelu(x):
    # return gelu_impl(x)
    return paddle.nn.functional.gelu(x, approximate=True)


class DalleTransformer(paddle.nn.Layer):
    """
    This module takes input from embedding layer and it's output can
    be used directly by a logit layer. It consists of L (num-layers)
    blocks of:
        layer norm
        self attention
        residual connection
        layer norm
        mlp
        residual connection
    followed by a final layer norm.

    Arguments:
        num_layers: Number of transformer layers.
        hidden_size: The hidden size of the self attention.
        num_attention_heads: number of attention head in the self
                             attention.
        attention_dropout_prob: dropout probability of the attention
                                score in self attention.
        output_dropout_prob: dropout probability for the outputs
                             after self attention and final output.
        layernorm_epsilon: epsilon used in layernorm to avoid
                           division by zero.
    """
    _mask_map = []

    def __init__(self, num_layers, hidden_size, num_attention_heads, attention_dropout_prob, output_dropout_prob,
                 layernorm_epsilon=1.0e-5, cogview_sandwich_layernorm=False, cogview_pb_relax=False):
        super(DalleTransformer, self).__init__()

        # CogView stabilization of training features, see chapter 2.4 https://arxiv.org/pdf/2105.13290.pdf
        self.cogview_pb_relax = cogview_pb_relax

        # Transformer layers.
        self.layers = paddle.nn.LayerList([
            DalleTransformerLayer(
                hidden_size,
                num_attention_heads,
                attention_dropout_prob,
                output_dropout_prob,
                layernorm_epsilon,
                cogview_sandwich_layernorm=cogview_sandwich_layernorm,
                cogview_pb_relax=cogview_pb_relax,
            ) for _ in range(num_layers)
        ])

        # Final layer norm before output.
        self.final_layernorm = LayerNorm(hidden_size, epsilon=layernorm_epsilon)

    def forward(self, hidden_states, attention_mask, has_cache, use_cache):
        for i, layer in enumerate(self.layers):
            mask = attention_mask
            if len(self._mask_map):
                layer_mask = self._mask_map[i][:mask.shape[2], :mask.shape[3]]
                mask = paddle.multiply(attention_mask, layer_mask)
            hidden_states, present_has_cache = layer(hidden_states, mask, has_cache=has_cache, use_cache=use_cache)
        output = self.final_layernorm(hidden_states)
        return output, present_has_cache


class DalleTransformerLayer(paddle.nn.Layer):
    """
    A single layer transformer.

    We use the following notation:
        h: hidden size
        n: number of attention heads
        b: batch size
        s: sequence length
    Transformer layer takes input with size [b, s, h] and returns an
    output of the same size.

    Arguments:
        hidden_size: The hidden size of the self attention.
        num_attention_heads: number of attention head in the self
                             attention.
        attention_dropout_prob: dropout probability of the attention
                                score in self attention.
        output_dropout_prob: dropout probability for the outputs
                             after self attention and final output.
        layernorm_epsilon: epsilon used in layernorm to avoid
                           division by zero.
    """

    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 attention_dropout_prob,
                 output_dropout_prob,
                 layernorm_epsilon,
                 cogview_sandwich_layernorm=False,
                 cogview_pb_relax=False):
        super(DalleTransformerLayer, self).__init__()

        # CogView stabilization of training features, see chapter 2.4 https://arxiv.org/pdf/2105.13290.pdf
        self.cogview_sandwich_layernorm = cogview_sandwich_layernorm
        self.cogview_pb_relax = cogview_pb_relax

        # Layernorm on the input data.
        self.input_layernorm = LayerNorm(hidden_size, epsilon=layernorm_epsilon)

        if self.cogview_sandwich_layernorm:
            self.before_first_addition_layernorm = LayerNorm(hidden_size, epsilon=layernorm_epsilon)
            self.before_second_addition_layernorm = LayerNorm(hidden_size, epsilon=layernorm_epsilon)

        # Self attention.
        self.attention = DalleSelfAttention(
            hidden_size,
            num_attention_heads,
            attention_dropout_prob,
            output_dropout_prob,
            cogview_pb_relax=cogview_pb_relax
        )

        # Layernorm on the input data.
        self.post_attention_layernorm = LayerNorm(hidden_size, epsilon=layernorm_epsilon)

        # MLP
        self.mlp = DalleMLP(hidden_size, output_dropout_prob)

    def forward(self, hidden_states, ltor_mask, has_cache, use_cache):
        # hidden_states: [b, s, h]
        # ltor_mask: [1, 1, s, s]

        # Layer norm at the begining of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)

        # Self attention.
        attention_output, att_has_cache = self.attention(
            layernorm_output, ltor_mask, has_cache=has_cache, use_cache=use_cache)

        if self.cogview_sandwich_layernorm:
            attention_output = self.before_first_addition_layernorm(attention_output)

        # Residual connection.
        layernorm_input = hidden_states + attention_output

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        # MLP.
        mlp_output, mlp_has_cache = self.mlp(
            layernorm_output, has_cache=has_cache, use_cache=use_cache)

        if self.cogview_sandwich_layernorm:
            mlp_output = self.before_second_addition_layernorm(mlp_output)

        # Second residual connection.
        output = layernorm_input + mlp_output

        return output, att_has_cache and mlp_has_cache


class DalleSelfAttention(paddle.nn.Layer):
    """
    Self-attention layer takes input with size [b, s, h] where b is
    the batch size, s is the sequence length, and h is the hidden size
    and creates output of the same size.
    Arguments:
        hidden_size: total hidden size of the layer (h).
        num_attention_heads: number of attention heads (n). Note that we
                             require n to be divisible by number of GPUs
                             used to parallelize the model. Also, we
                             require hidden size to be divisible by n.
        attention_dropout_prob: dropout probability for the attention scores.
        output_dropout_prob: dropout probability for the output.
    We use the following notation:
        h: hidden_size
        n: num_attention_heads
        p: number of partitions
        np: n/p
        hp: h/p
        hn: h/n
        b: batch size
        s: sequence length
    """

    def __init__(self, hidden_size, num_attention_heads,
                 attention_dropout_prob, output_dropout_prob, cogview_pb_relax=False):
        super(DalleSelfAttention, self).__init__()

        # CogView stabilization of training features, see chapter 2.4 https://arxiv.org/pdf/2105.13290.pdf
        self.cogview_pb_relax = cogview_pb_relax

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.hidden_size_per_attention_head = divide(hidden_size, num_attention_heads)

        self.query_key_value = paddle.nn.Linear(hidden_size, 3*hidden_size)
        self.attention_dropout = paddle.nn.Dropout(attention_dropout_prob)

        # Output.
        self.dense = paddle.nn.Linear(hidden_size, hidden_size)
        self.output_dropout = paddle.nn.Dropout(output_dropout_prob)

        # Cache
        self.past_key = None
        self.past_value = None
        self.past_output = None

    def _transpose_for_scores(self, tensor):
        """ Transpose a 3D tensor [b, s, np*hn] into a 4D tensor with size [b, np, s, hn]. """
        new_tensor_shape = tensor.shape[:-1] + [self.num_attention_heads, self.hidden_size_per_attention_head]
        tensor = tensor.reshape(new_tensor_shape)
        return tensor.transpose([0, 2, 1, 3])

    def _calculate_attention_scores(self, query_layer, key_layer, ltor_mask):
        key_t = key_layer.swapaxes(-1, -2)
        if self.cogview_pb_relax:
            attention_scores = paddle.matmul(
                query_layer / math.sqrt(self.hidden_size_per_attention_head),
                key_t
            )
        else:
            attention_scores = paddle.matmul(query_layer, key_t) / math.sqrt(self.hidden_size_per_attention_head)
        ltor_mask = ltor_mask[:, :, -attention_scores.shape[-2]:]
        attention_scores = paddle.multiply(attention_scores, ltor_mask) - 10000.0 * (1.0 - ltor_mask)
        if self.cogview_pb_relax:
            # normalize attention scores. Should not affect resulting softmax value
            alpha = 32
            attention_scores_scaled = attention_scores / alpha
            attention_scores_scaled_maxes = attention_scores_scaled.detach().reshape(
                [attention_scores.shape[0], attention_scores.shape[1], -1]
            ).max(axis=-1)  # max per head per sample
            attention_scores_scaled_maxes = attention_scores_scaled_maxes.unsqueeze(-1).unsqueeze(-1).expand(
                [-1, -1, attention_scores.shape[2], attention_scores.shape[3]]
            )  # expand to [b, np, s, s]
            attention_scores = (attention_scores_scaled - attention_scores_scaled_maxes) * alpha
        return attention_scores

    def forward(self, hidden_states, ltor_mask, has_cache=False, use_cache=False,):
        # hidden_states: [b, s, h]
        # ltor_mask: [1, 1, s, s]
        # Attention heads. [b, s, hp]
        if has_cache and use_cache:
            mixed_x_layer = self.query_key_value(hidden_states[:, -1:, :])
        else:
            mixed_x_layer = self.query_key_value(hidden_states)

        (mixed_query_layer,
         mixed_key_layer,
         mixed_value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)

        query_layer = self._transpose_for_scores(mixed_query_layer)
        key_layer = self._transpose_for_scores(mixed_key_layer)
        value_layer = self._transpose_for_scores(mixed_value_layer)

        # Can be simplified, but I didn't for readability's sake
        if use_cache and has_cache:
            key_layer = paddle.concat((self.past_key, key_layer), axis=-2)
            value_layer = paddle.concat((self.past_value, value_layer), axis=-2)
            attention_scores = self._calculate_attention_scores(
                query_layer=query_layer, key_layer=key_layer, ltor_mask=ltor_mask
            )
        else:
            attention_scores = self._calculate_attention_scores(
                query_layer=query_layer, key_layer=key_layer, ltor_mask=ltor_mask
            )

        if use_cache:
            self.past_key = key_layer
            self.past_value = value_layer
        else:
            self.past_key = None
            self.past_value = None
            self.past_output = None
            has_cache = False

        if use_cache and has_cache:
            attention_scores = attention_scores[..., -1:, :]

        # Attention probabilities. [b, np, s, s]
        attention_probs = paddle.nn.Softmax(axis=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attention_dropout(attention_probs)

        # Context layer.
        # [b, np, s, hn]
        context_layer = paddle.matmul(attention_probs, value_layer)

        # [b, s, np, hn]
        context_layer = context_layer.transpose([0, 2, 1, 3])

        new_context_layer_shape = context_layer.shape[:-2] + [self.hidden_size, ]
        # [b, s, hp]
        context_layer = context_layer.reshape(new_context_layer_shape)

        # Output. [b, s, h]
        output = self.dense(context_layer)

        if use_cache:
            # Can be simplified, but I didn't for readability's sake
            if has_cache:
                output = paddle.concat((self.past_output, output), axis=-2)
                self.past_output = output
            else:
                self.past_output = output
            has_cache = True

        output = self.output_dropout(output)
        return output, has_cache


class DalleMLP(paddle.nn.Layer):
    """
    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform gelu transformation, and project the
    state back into h hidden dimension. At the end, dropout is also
    applied.
    Arguments:
        hidden_size: The hidden size of the self attention.
        output_dropout_prob: dropout probability for the outputs
                             after self attention and final output.
    """

    def __init__(self, hidden_size, output_dropout_prob):
        super(DalleMLP, self).__init__()
        # Project to 4h.
        self.dense_h_to_4h = paddle.nn.Linear(hidden_size, 4*hidden_size)
        # Project back to h.
        self.dense_4h_to_h = paddle.nn.Linear(4*hidden_size, hidden_size)
        self.dropout = paddle.nn.Dropout(output_dropout_prob)
        # MLP cache
        self.past_x = None

    def forward(self, hidden_states, has_cache=False, use_cache=False):
        if has_cache and use_cache:
            hidden_states = hidden_states[:, -1:]

        # [b, s, 4hp]
        x = self.dense_h_to_4h(hidden_states)
        x = gelu(x)
        # [b, s, h]
        x = self.dense_4h_to_h(x)
        if use_cache:
            # Can be simplified, but I didn't for readability's sake
            if has_cache:
                x = paddle.concat((self.past_x, x), axis=-2)
                self.past_x = x
            else:
                self.past_x = x

            has_cache = True
        else:
            self.past_x = None
            has_cache = False
        output = self.dropout(x)

        return output, has_cache
