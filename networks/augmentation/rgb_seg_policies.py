

import itertools
import numpy as np
import torch
import math
from networks.augmentation.img_ops.autosegment import ShearX, ShearY, Color, Invert, Solarize, TranslateX, TranslateY, Rotate, AutoContrast, Equalize, Posterize, Contrast, Brightness, Sharpness

all_augmentations = [
    ShearX, # 1 good
    ShearY, # 2
    TranslateX, # 3 good
    TranslateY, # 4
    
    Color,
    Contrast, #11
    Brightness, #13
    Sharpness, #14
    
    # Invert,
    
    # Rotate, # 5
    AutoContrast, #6
    Equalize, #8 
    Posterize, #10,

    # Solarize
]

IMG_OPS = {
    'Identity': [0.0 , 1.0],
    'ShearX': [-0.3, 0.3],  # 0
    'ShearY': [-0.3, 0.3],  # 1
    'TranslateX': [-0.45, 0.45],  # 2
    'TranslateY': [-0.45, 0.45],  # 3
    'Rotate': [-math.pi/12, -math.pi/12],  # 4
    'AutoContrast': [0, 1],  # 5
    'Invert': [0, 1],  # 6
    'Equalize': [0, 1],  # 7
    'Solarize': [0, 1],  # 8
    'Posterize': [4, 8],  # 9
    'Contrast': [0.1, 1.9],  # 10
    'Color': [0.1, 1.9],  # 11
    'Brightness': [0.1, 1.9],  # 12
    'Sharpness': [0.1, 1.9],  # 13
    'GaussianBlur': [0.5, 2.0],  # 14
}


def get_sub_policies(n_aug=-1):
    sub_policies = None
    all_augmentations_index = list(range(len(all_augmentations)))
    if n_aug == -1: # all possible combinations 2^N
        sub_policies = list(itertools.chain.from_iterable(itertools.combinations(all_augmentations_index, n) for n in range(1, len(all_augmentations_index) + 1)))
    else:
        sub_policies = list(itertools.combinations(all_augmentations_index, n_aug))

    return sub_policies


def apply_augment(x, fn_idx, mag):
    # Image ops expects images in the range 0 and 1.
    x = torch.clamp(x, 0.0, 1.0)
    fn = all_augmentations[fn_idx]
    min_val, max_val = IMG_OPS[fn.__name__]
    v = min_val + mag * (max_val - min_val)
    
    out, ctx = fn(x, v)
    out = torch.clamp(out, 0.0, 1.0)
    return out, ctx
