# -*- coding: utf-8 -*-
import io
from os.path import abspath, dirname

import PIL
import pytest
import requests

from rudalle_paddle import get_tokenizer, get_rudalle_model, get_vae, get_realesrgan


TEST_ROOT = dirname(abspath(__file__))


@pytest.fixture(scope='module')
def realesrgan():
    realesrgan = get_realesrgan('x2-paddle', device='cpu')
    yield realesrgan


@pytest.fixture(scope='module')
def vae():
    vae = get_vae(pretrained=False)
    yield vae


@pytest.fixture(scope='module')
def yttm_tokenizer():
    tokenizer = get_tokenizer()
    yield tokenizer


@pytest.fixture(scope='module')
def sample_image():
    url = 'https://cdn.kqed.org/wp-content/uploads/sites/12/2013/12/rudolph.png'
    resp = requests.get(url)
    resp.raise_for_status()
    image = PIL.Image.open(io.BytesIO(resp.content))
    yield image


@pytest.fixture(scope='module')
def small_dalle():
    model = get_rudalle_model('small', pretrained=False, fp16=False, device='cpu')
    return model
