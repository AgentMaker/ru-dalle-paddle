# -*- coding: utf-8 -*-
from math import sqrt, log

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import einsum
from einops import rearrange
from taming.modules.diffusionmodules.model import Encoder, Decoder


class VQGanGumbelVAE(paddle.nn.Layer):

    def __init__(self, config):
        super().__init__()
        model = GumbelVQ(
            ddconfig=config.model.params.ddconfig,
            n_embed=config.model.params.n_embed,
            embed_dim=config.model.params.embed_dim,
            kl_weight=config.model.params.kl_weight,
        )
        self.model = model
        self.num_layers = int(log(config.model.params.ddconfig.attn_resolutions[0]) / log(2))
        self.image_size = 256
        self.num_tokens = config.model.params.n_embed

    @paddle.no_grad()
    def get_codebook_indices(self, img):
        img = (2 * img) - 1
        _, _, [_, _, indices] = self.model.encode(img)
        return rearrange(indices, 'b h w -> b (h w)')

    def decode(self, img_seq):
        b, n = img_seq.shape
        one_hot_indices = paddle.nn.functional.one_hot(img_seq, num_classes=self.num_tokens).astype(paddle.float32)
        z = paddle.matmul(one_hot_indices, self.model.quantize.embed.weight)
        z = rearrange(z, 'b (h w) c -> b c h w', h=int(sqrt(n)))
        img = self.model.decode(z)
        img = (img.clip(-1., 1.) + 1) * 0.5
        return img

    @staticmethod
    def convert(model_path):
        import os
        import torch
        torch_weights = model_path
        target_model_path = model_path[:-5] + '.pdckpt'
        if os.path.exists(target_model_path):
            return target_model_path
        state_dict = torch.load(torch_weights, map_location='cpu')['state_dict']

        paddle_state_dict = {}
        for name, param in state_dict.items():
            if 'dense' in name and param.ndim == 2:
                param = param.transpose(1, 0)
            if param.ndim == 0:
                param = param.unsqueeze(0)
            param = param.cpu().detach().numpy()
            paddle_state_dict[name] = param

        paddle.save({'state_dict': paddle_state_dict}, target_model_path)

        return model_path


class GumbelQuantize(nn.Layer):
    """
    credit to @karpathy: https://github.com/karpathy/deep-vector-quantization/blob/main/model.py (thanks!)
    Gumbel Softmax trick quantizer
    Categorical Reparameterization with Gumbel-Softmax, Jang et al. 2016
    https://arxiv.org/abs/1611.01144
    """

    def __init__(self, num_hiddens, embedding_dim, n_embed, straight_through=True,
                 kl_weight=5e-4, temp_init=1.0, use_vqinterface=True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_embed = n_embed
        self.straight_through = straight_through
        self.temperature = temp_init
        self.kl_weight = kl_weight
        self.proj = nn.Conv2D(num_hiddens, n_embed, 1)
        self.embed = nn.Embedding(self.n_embed, self.embedding_dim)
        self.use_vqinterface = use_vqinterface

    def forward(self, z, temp=None, return_logits=False):
        hard = self.straight_through if self.training else True
        temp = self.temperature if temp is None else temp
        logits = self.proj(z)
        soft_one_hot = F.gumbel_softmax(logits, tau=temp, axis=1, hard=hard)
        z_q = einsum('b n h w, n d -> b d h w', soft_one_hot, self.embed.weight)
        # + kl divergence to the prior loss
        qy = F.softmax(logits, axis=1)
        diff = self.kl_weight * paddle.sum(qy * paddle.log(qy * self.n_embed + 1e-10), axis=1).mean()
        ind = soft_one_hot.argmax(axis=1)
        if self.use_vqinterface:
            if return_logits:
                return z_q, diff, (None, None, ind), logits
            return z_q, diff, (None, None, ind)
        return z_q, diff, ind


class GumbelVQ(nn.Layer):

    def __init__(self, ddconfig, n_embed, embed_dim, kl_weight=1e-8):
        super().__init__()
        z_channels = ddconfig['z_channels']
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quantize = GumbelQuantize(z_channels, embed_dim, n_embed=n_embed, kl_weight=kl_weight, temp_init=1.0)
        self.quant_conv = paddle.nn.Conv2D(ddconfig['z_channels'], embed_dim, 1)
        self.post_quant_conv = paddle.nn.Conv2D(embed_dim, ddconfig['z_channels'], 1)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec
