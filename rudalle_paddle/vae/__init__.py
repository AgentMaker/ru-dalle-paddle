# -*- coding: utf-8 -*-
from os.path import dirname, abspath, join

import paddle
from huggingface_hub import hf_hub_url, cached_download
from omegaconf import OmegaConf

from .model import VQGanGumbelVAE


MODELS = {
    'vqgan.gumbelf8-sber': dict(
        repo_id='shonenkov/rudalle-utils',
        filename='vqgan.gumbelf8-sber.model.ckpt',
    ),
    'vqgan.gumbelf8-sber.paddle': dict(
        repo_id='HighCWu/rudalle-paddle-utils',
        filename='vqgan.gumbelf8-sber.model.pdckpt',
    ),
}


def get_vae(name='vqgan.gumbelf8-sber.paddle', pretrained=True, cache_dir='/tmp/rudalle'):
    # TODO
    config = OmegaConf.load(join(dirname(abspath(__file__)), 'vqgan.gumbelf8-sber.config.yml'))
    vae = VQGanGumbelVAE(config)
    config = MODELS[name]
    if pretrained:
        cache_dir = join(cache_dir, name)
        config_file_url = hf_hub_url(repo_id=config['repo_id'], filename=config['filename'])
        cached_download(config_file_url, cache_dir=cache_dir, force_filename=config['filename'])
        if config['filename'][-5:] == '.ckpt':
            VQGanGumbelVAE.convert(join(cache_dir, config['filename']))
            config['filename'] = config['filename'][:-5] + '.pdckpt'
        checkpoint = paddle.load(join(cache_dir, config['filename']))
        vae.model.set_state_dict(checkpoint['state_dict'])
    print('vae --> ready')
    return vae
