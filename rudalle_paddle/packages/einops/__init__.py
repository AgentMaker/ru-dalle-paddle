# -*- coding: utf-8 -*-
__author__ = 'Alex Rogozhnikov'
__version__ = '0.3.2'


from .einops import rearrange, reduce, repeat, parse_shape, asnumpy, EinopsError


__all__ = ['rearrange', 'reduce', 'repeat', 'parse_shape', 'asnumpy', 'EinopsError']
