import img_ops.autoaugment as autoaugment
import math
import numpy
import torch

import os
from PIL import Image

def test_autoaugment(img_path, out_path):
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    m = 0.07

    img = numpy.array(Image.open(img_path)).astype(numpy.float32)

    img = img / 255.0

    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

    augment_list = [
        autoaugment.ShearX,
        autoaugment.ShearY,
        autoaugment.TranslateX,
        autoaugment.TranslateY,
        autoaugment.Rotate,
        autoaugment.AutoContrast,
        autoaugment.Invert,
        autoaugment.Equalize,
        autoaugment.Solarize,
        autoaugment.Posterize,
        autoaugment.Contrast,
        autoaugment.Color,
        autoaugment.Brightness,
        autoaugment.Sharpness,
        autoaugment.GaussianBlur
    ]

    augment_mag = {
        'ShearX': [-0.3, 0.3],  # 0
        'ShearY': [-0.3, 0.3],  # 1
        'TranslateX': [-0.45, 0.45],  # 2
        'TranslateY': [-0.45, 0.45],  # 3
        'Rotate': [-math.pi/6, -math.pi/6],  # 4
        'AutoContrast': [0, 1],  # 5
        'Invert': [0, 1],  # 6
        'Equalize': [0, 1],  # 7
        'Solarize': [0, 1],  # 8
        'Posterize': [4, 8],  # 9
        'Contrast': [0.1, 1.9],  # 10
        'Color': [0.1, 1.9],  # 11
        'Brightness': [0.1, 1.9],  # 12
        'Sharpness': [0.1, 1.9],  # 13
        'GaussianBlur': [1.0, 5.0],  # 14
    }


    for fn in augment_list:
        print("Testing %s" %(fn.__name__))
        min_val, max_val = augment_mag[fn.__name__]
        v = torch.tensor(m * (max_val - min_val) + min_val).unsqueeze(0)

        img.requires_grad_(True)
        v.requires_grad_(True)
        
        aug_img = fn(img, v)
        aug_img = torch.clamp(aug_img, 0, 1)

        # check backward branch
        loss = aug_img.sum()
        loss.backward()

        # save img
        aug_numpy_img = aug_img.detach()[0].cpu().permute(1, 2, 0).numpy()
        aug_numpy_img = 255.0*aug_numpy_img
        aug_numpy_img = aug_numpy_img.astype(numpy.uint8)

        Image.fromarray(aug_numpy_img).save(os.path.join(out_path, fn.__name__ + '.png'))

