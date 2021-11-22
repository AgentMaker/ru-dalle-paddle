# -*- coding: utf-8 -*-
import os
import random

import paddle
import numpy as np

from PIL import Image


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def paddle_tensors_to_pil_list(input_images):
    out_images = []
    for in_image in input_images:
        in_image = np.clip(in_image.cpu().detach().numpy() * 255, 0, 255)
        out_image = Image.fromarray(in_image.transpose([1, 2, 0]).astype(np.uint8))
        out_images.append(out_image)
    return out_images


def pil_list_to_paddle_tensors(pil_images):
    result = []
    for pil_image in pil_images:
        image = np.array(pil_image, dtype=np.uint8)
        image = paddle.to_tensor(image).astype(paddle.int64)
        image = image.transpose([2, 0, 1]).unsqueeze(0)
        result.append(image)
    return paddle.concat(result, axis=0)
