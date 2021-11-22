# -*- coding: utf-8 -*-
import os
import glob
import shutil

from rudalle_paddle import get_rudalle_model, get_tokenizer, get_vae, get_realesrgan, get_ruclip

target_dir = 'rudalle-paddle-utils'
os.makedirs(target_dir, exist_ok=True)

get_rudalle_model('Malevich', True, device='cuda', cache_dir='/tmp/rudalle')
get_tokenizer(cache_dir='/tmp/rudalle')
get_vae(name='vqgan.gumbelf8-sber', pretrained=True, cache_dir='/tmp/rudalle')
get_realesrgan('x2', cache_dir='/tmp/rudalle')
get_realesrgan('x4', cache_dir='/tmp/rudalle')
get_realesrgan('x8', cache_dir='/tmp/rudalle')

files = glob.glob('/tmp/rudalle/**/*.pkl', recursive=True) + \
    glob.glob('/tmp/rudalle/**/*.json', recursive=True) + \
    glob.glob('/tmp/rudalle/**/*.model', recursive=True) + \
    glob.glob('/tmp/rudalle/**/*.pdckpt', recursive=True) + \
    glob.glob('/tmp/rudalle/**/*.pdparams', recursive=True)

for path in files:
    name = os.path.basename(path)
    shutil.move(path, os.path.join(target_dir, name))

target_dir = os.path.join(target_dir, 'ruclip-vit-base-patch32-v5-paddle')
os.makedirs(target_dir, exist_ok=True)

get_ruclip('ruclip-vit-base-patch32-v5', cache_dir='/tmp')

files = glob.glob('/tmp/ruclip-vit-base-patch32-v5/**/*.pkl', recursive=True) + \
    glob.glob('/tmp/ruclip-vit-base-patch32-v5/**/*.json', recursive=True) + \
    glob.glob('/tmp/ruclip-vit-base-patch32-v5/**/*.model', recursive=True) + \
    glob.glob('/tmp/ruclip-vit-base-patch32-v5/**/*.pdckpt', recursive=True) + \
    glob.glob('/tmp/ruclip-vit-base-patch32-v5/**/*.pdparams', recursive=True)

for path in files:
    name = os.path.basename(path)
    shutil.move(path, os.path.join(target_dir, name))
