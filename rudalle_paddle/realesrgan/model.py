# -*- coding: utf-8 -*-
# Source: https://github.com/boomb0om/Real-ESRGAN-colab

import paddle
import numpy as np
from PIL import Image

from .rrdbnet_arch import RRDBNet
from .utils import pad_reflect, split_image_into_overlapping_patches, stich_together, unpad_image


class RealESRGAN:
    def __init__(self, device, scale=4):
        self.device = device
        self.scale = scale
        self.model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)

    def load_weights(self, model_path):
        if model_path[-4:] == '.pth':
            model_path = RealESRGAN.convert(model_path)
        loadnet = paddle.load(model_path)
        if 'params' in loadnet:
            self.model.set_state_dict(loadnet['params'])
        elif 'params_ema' in loadnet:
            self.model.set_state_dict(loadnet['params_ema'])
        else:
            self.model.set_state_dict(loadnet)
        self.model.eval()
        self.model.to(self.device)

    def predict(self, lr_image, batch_size=4, patches_size=192,
                padding=24, pad_size=15):
        scale = self.scale
        device = self.device
        lr_image = np.array(lr_image)
        lr_image = pad_reflect(lr_image, pad_size)

        patches, p_shape = split_image_into_overlapping_patches(lr_image, patch_size=patches_size,
                                                                padding_size=padding)
        img = paddle.to_tensor(patches / 255.).astype(paddle.float32).transpose((0, 3, 1, 2)).to(device).detach()

        with paddle.no_grad():
            res = self.model(img[0:batch_size])
            for i in range(batch_size, img.shape[0], batch_size):
                res = paddle.concat((res, self.model(img[i:i + batch_size])), 0)

        sr_image = res.transpose((0, 2, 3, 1)).cpu().clip_(0, 1)
        np_sr_image = sr_image.numpy()

        padded_size_scaled = tuple(np.multiply(p_shape[0:2], scale)) + (3,)
        scaled_image_shape = tuple(np.multiply(lr_image.shape[0:2], scale)) + (3,)
        np_sr_image = stich_together(np_sr_image, padded_image_shape=padded_size_scaled,
                                     target_shape=scaled_image_shape, padding_size=padding * scale)
        sr_img = (np_sr_image * 255).astype(np.uint8)
        sr_img = unpad_image(sr_img, pad_size * scale)
        sr_img = Image.fromarray(sr_img)

        return sr_img

    @staticmethod
    def convert(model_path):
        import os
        import torch
        torch_weights = model_path
        target_model_path = model_path[:-4] + '.pdparams'
        if os.path.exists(target_model_path):
            return target_model_path
        _state_dict = torch.load(torch_weights, map_location='cpu')
        if 'params' in _state_dict:
            state_dict = _state_dict['params']
        elif 'params_ema' in _state_dict:
            state_dict = _state_dict['params_ema']
        else:
            state_dict = _state_dict

        paddle_state_dict = {}
        for name, param in state_dict.items():
            if param.ndim == 0:
                param = param.unsqueeze(0)
            param = param.cpu().detach().numpy()
            paddle_state_dict[name] = param

        if 'params' in _state_dict:
            paddle_state_dict = {'params': paddle_state_dict}
        elif 'params_ema' in _state_dict:
            paddle_state_dict = {'params_ema': paddle_state_dict}

        paddle.save(paddle_state_dict, target_model_path)

        return target_model_path
