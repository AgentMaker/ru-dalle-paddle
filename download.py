# -*- coding: utf-8 -*-
from rudalle_paddle import get_rudalle_model, get_tokenizer, get_vae, get_realesrgan, get_ruclip

get_rudalle_model('Malevich-paddle', True, device='cuda', cache_dir='pretrained_models')
get_tokenizer(cache_dir='pretrained_models')
get_vae(name='vqgan.gumbelf8-sber.paddle', pretrained=True, cache_dir='pretrained_models')
get_realesrgan('x2-paddle', cache_dir='pretrained_models')
get_realesrgan('x4-paddle', cache_dir='pretrained_models')
get_realesrgan('x8-paddle', cache_dir='pretrained_models')
get_ruclip('ruclip-vit-base-patch32-v5-paddle', cache_dir='pretrained_models')
