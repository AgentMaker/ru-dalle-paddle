# -*- coding: utf-8 -*-
import os

import pickle
import paddle
from huggingface_hub import hf_hub_url, cached_download

from .model import DalleModel
from .fp16 import FP16Module


MODELS = {
    'Malevich': dict(
        description='◼️ Malevich is 1.3 billion params model from the family GPT3-like, '
                    'that uses Russian language and text+image multi-modality.',
        model_params=dict(
            num_layers=24,
            hidden_size=2048,
            num_attention_heads=16,
            embedding_dropout_prob=0.1,
            output_dropout_prob=0.1,
            attention_dropout_prob=0.1,
            image_tokens_per_dim=32,
            text_seq_length=128,
            use_masks=True,
            cogview_sandwich_layernorm=True,
            cogview_pb_relax=True,
            vocab_size=16384+128,
            image_vocab_size=8192,
        ),
        repo_id='sberbank-ai/rudalle-Malevich',
        filename='pytorch_model.bin',
        full_description='',  # TODO
    ),
    'Malevich-paddle': dict(
        description='◼️ Malevich is 1.3 billion params model from the family GPT3-like, '
                    'that uses Russian language and text+image multi-modality.',
        model_params=dict(
            num_layers=24,
            hidden_size=2048,
            num_attention_heads=16,
            embedding_dropout_prob=0.1,
            output_dropout_prob=0.1,
            attention_dropout_prob=0.1,
            image_tokens_per_dim=32,
            text_seq_length=128,
            use_masks=True,
            cogview_sandwich_layernorm=True,
            cogview_pb_relax=True,
            vocab_size=16384+128,
            image_vocab_size=8192,
        ),
        repo_id='HighCWu/rudalle-paddle-utils',
        filename='rudalle_paddle.pkl',
        full_description='',  # TODO
    ),
    'small': dict(
        description='',
        model_params=dict(
            num_layers=12,
            hidden_size=768,
            num_attention_heads=12,
            embedding_dropout_prob=0.1,
            output_dropout_prob=0.1,
            attention_dropout_prob=0.1,
            image_tokens_per_dim=32,
            text_seq_length=128,
            use_masks=True,
            cogview_sandwich_layernorm=True,
            cogview_pb_relax=True,
            vocab_size=16384+128,
            image_vocab_size=8192,
        ),
        repo_id='',
        filename='',
        full_description='',  # TODO
    ),
}


def get_rudalle_model(name, pretrained=True, fp16=False, device='cpu', cache_dir='/tmp/rudalle'):
    # TODO docstring
    assert name in MODELS

    paddle.set_device(device)

    if fp16 and device == 'cpu':
        print('Warning! Using both fp16 and cpu doesnt support. You can use cuda device or turn off fp16.')

    config = MODELS[name]
    model = DalleModel(device=device, fp16=fp16, **config['model_params'])
    if pretrained:
        cache_dir = os.path.join(cache_dir, name)
        config_file_url = hf_hub_url(repo_id=config['repo_id'], filename=config['filename'])
        cached_download(config_file_url, cache_dir=cache_dir, force_filename=config['filename'])
        if config['filename'] == 'pytorch_model.bin':
            DalleModel.convert(cache_dir)
            config['filename'] = 'rudalle_paddle.pkl'
        with open(os.path.join(cache_dir, config['filename']), 'rb') as f:
            # paddle.load could not load large paddle.save file
            checkpoint = pickle.load(f)
        checkpoint = {k: v.astype('float32') for k, v in checkpoint.items()}
        model.set_state_dict(checkpoint)
    if fp16:
        model = FP16Module(model)
    model.eval()
    model.to(device)
    if config['description'] and pretrained:
        print(config['description'])
    return model
