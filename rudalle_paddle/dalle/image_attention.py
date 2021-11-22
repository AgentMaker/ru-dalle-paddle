# -*- coding: utf-8 -*-

import paddle


def _init_mask(text_tokens, image_tokens_per_dim):
    attn_size = text_tokens + image_tokens_per_dim**2
    mask = paddle.tril(paddle.ones([attn_size, attn_size]))
    return mask


def get_row_mask(text_tokens=256, image_tokens_per_dim=32):
    mask = _init_mask(text_tokens, image_tokens_per_dim)
    step = image_tokens_per_dim + 1
    for col in range(text_tokens, mask.shape[1]):
        if col + step >= mask.shape[0] or col > mask.shape[1]:
            continue  # paddle not support index >= tensor shape
        mask[col + step:, col] = 0.0
    return mask


def get_col_mask(text_tokens=256, image_tokens_per_dim=32):
    mask = _init_mask(text_tokens, image_tokens_per_dim)
    step = image_tokens_per_dim - 1
    for col in range(text_tokens, mask.shape[1]):
        for i in range(1, mask.shape[0], step+1):
            if col + i >= mask.shape[0] or col + i + step > mask.shape[0] or col > mask.shape[1]:
                continue  # paddle not support index >= tensor shape
            mask[col + i: col + i + step, col] = 0.0
    return mask


def get_conv_mask(text_tokens=256, image_tokens_per_dim=32, kernel=11):
    mask = _init_mask(text_tokens, image_tokens_per_dim)
    shift = kernel // 2
    for pos in range(text_tokens, mask.shape[1]):
        if pos + 1 < mask.shape[0] and pos < mask.shape[1]:
            mask[pos+1:, pos] = 0.0
        img = paddle.zeros([image_tokens_per_dim, image_tokens_per_dim])
        pixel_id = pos - text_tokens
        row = pixel_id // image_tokens_per_dim
        col = pixel_id % image_tokens_per_dim
        for r in range(-shift, shift+1):
            for c in range(-shift, shift+1):
                c_abs = (c + col) % image_tokens_per_dim
                r_abs = (r + row) % image_tokens_per_dim
                img[r_abs, c_abs] = 0.2
                cell_id = r_abs * image_tokens_per_dim + c_abs
                if text_tokens + cell_id > pos:
                    mask[text_tokens + cell_id, pos] = 1.0

        img[row, col] = 1.0
    return mask
