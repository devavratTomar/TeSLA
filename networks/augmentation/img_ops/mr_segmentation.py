import torch
from .constants import GAUSSIAN_KERNEL

"""
Assumption. Image intensities are between 0 and 1
Image shape is batch x ch x h x w
Value shape is batch
"""

"""
HELPER FUNCTIONS
"""

def apply_invert_affine(x, affine):
    # affine shape should be batch x 2 x 3
    # x shape should be batch x ch x h x w

    # get homomorphic transform
    H = torch.nn.functional.pad(affine, [0, 0, 0, 1], "constant", value=0.0)
    H[..., -1, -1] += 1.0

    inv_H = torch.inverse(H)
    inv_affine = inv_H[:, :2, :3]

    grid = torch.nn.functional.affine_grid(inv_affine, x.size(), align_corners=False)
    x = torch.nn.functional.grid_sample(x, grid, padding_mode="reflection", align_corners=False)

    return x

def apply_affine(x, affine):
    grid = torch.nn.functional.affine_grid(affine, x.size(), align_corners=False)
    x = torch.nn.functional.grid_sample(x, grid, padding_mode="reflection", align_corners=False)

    return x


def blend(img1, img2, factor):
    """
    blends img1 and img2 using factor. factor 0 implies only img1 is used and 1 implies only img2 is used.
    img1 shape should be batch, ch, h, w
    img2 shape should be batch, ch, h, w
    factor shape should be batch
    """
    factor = factor.reshape(-1, 1, 1, 1) # add dim of ch, h, w
    diff = img2 - img1
    scaled = factor * diff
    tmp = img1 + scaled
    # return tmp
    return torch.clamp(tmp, 0.0, 1.0)

"""
AUGMENTATIONS: return aug img, * needed for inverting spatial deformation
"""


def Identity(x, v):
    affine = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32, device=x.device).reshape(-1, 2, 3).repeat(x.size(0), 1, 1)
    v = v.reshape(-1, 1, 1, 1)
    return x, affine


def Gamma(x, v):
    affine = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32, device=x.device).reshape(-1, 2, 3).repeat(x.size(0), 1, 1)
    v = v.reshape(-1, 1, 1, 1)
    x = x ** v
    return x, affine


def Brightness(x, v):
    affine = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32, device=x.device).reshape(-1, 2, 3).repeat(x.size(0), 1, 1)
    degenerate = torch.zeros_like(x)
    return blend(degenerate, x, v), affine


def Contrast(x, v):
    affine = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32, device=x.device).reshape(-1, 2, 3).repeat(x.size(0), 1, 1)
    mean_img = torch.mean(x, dim=(1, 2, 3), keepdim=True)
    degenerate = torch.zeros_like(x) + mean_img
    return blend(degenerate, x, v), affine


def GaussianBlur(x, v):
    affine = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32, device=x.device).reshape(-1, 2, 3).repeat(x.size(0), 1, 1)
    v = v.reshape(-1, 1, 1, 1, 1)

    gauss_kernel = GAUSSIAN_KERNEL.to(x.device)[None, ...]**(1/(v**2))
    w = gauss_kernel/gauss_kernel.sum(dim=(1, 2, 3, 4), keepdim=True)

    x = torch.nn.functional.pad(x, (3, 3, 3, 3), mode='replicate')

    batch_size = x.size(0)
    o = torch.nn.functional.conv2d(
        x.view(1, batch_size*1, x.size(2), x.size(3)),
        w.view(batch_size*1, 1, w.size(3), w.size(4)),
        groups=batch_size)
    
    o = o.view(batch_size, 1, o.size(2), o.size(3))

    return o, affine


def RandomResizeCrop(x, v):
    delta_scale_x = torch.abs(v)
    delta_scale_y = torch.abs(v)

    scale_matrix_x = torch.tensor([1, 0, 0, 0, 0, 0], dtype=torch.float32, device=x.device).reshape(-1, 2, 3)
    scale_matrix_y = torch.tensor([0, 0, 0, 0, 1, 0], dtype=torch.float32, device=x.device).reshape(-1, 2, 3)

    translation_matrix_x = torch.tensor([0, 0, 1, 0, 0, 0], dtype=torch.float32, device=x.device).reshape(-1, 2, 3)
    translation_matrix_y = torch.tensor([0, 0, 0, 0, 0, 1], dtype=torch.float32, device=x.device).reshape(-1, 2, 3)

    delta_x = 0.5 * delta_scale_x * (2*torch.rand(x.size(0), 1, 1, device=x.device) - 1.0)
    delta_y = 0.5 * delta_scale_y * (2*torch.rand(x.size(0), 1, 1, device=x.device) -1.0)

    random_affine = (1 - delta_scale_x) * scale_matrix_x + (1 - delta_scale_y) * scale_matrix_y +\
                delta_x * translation_matrix_x + \
                delta_y * translation_matrix_y

    x = apply_affine(x, random_affine)
    return x, random_affine.detach()


def RandomHorizontalFlip(x, v):
    affine = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32, device=x.device).reshape(-1, 2, 3).repeat(x.size(0), 1, 1)
    horizontal_flip = torch.tensor([-1, 0, 0, 0, 1, 0], dtype=torch.float32, device=x.device).reshape(-1, 2, 3)
    # randomly flip some of the images in the batch
    mask = (torch.rand(x.size(0), device=x.device) > 0.5)
    # mask = v > 0.5
    affine[mask] = affine[mask] * horizontal_flip
    v = v.view(-1, 1, 1, 1)
    x = apply_affine(x, affine)
    return x, affine.detach()


def RandomVerticalFlip(x, v):
    affine = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32, device=x.device).reshape(-1, 2, 3).repeat(x.size(0), 1, 1)
    vertical_flip = torch.tensor([1, 0, 0, 0, -1, 0], dtype=torch.float32, device=x.device).reshape(-1, 2, 3)
    # randomly flip some of the images in the batch
    
    mask = (torch.rand(x.size(0), device=x.device) > 0.5)
    # mask = v > 0.5
    affine[mask] = affine[mask] * vertical_flip

    v = v.view(-1, 1, 1, 1)
    x = apply_affine(x, affine) #+  v - v.detach()
    return x, affine.detach()


def RandomRotate(x, v):
    # affine = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32, device=x.device).reshape(-1, 2, 3).repeat(x.size(0), 1, 1)
    # rotation = torch.tensor([0, -1, 0, 1, 0, 0], dtype=torch.float32, device=x.device).reshape(-1, 2, 3)
    # # randomly flip some of the images in the batch
    # # mask = (torch.rand(x.size(0), device=x.device) < v)
    
    # mask = v > 0.5
    # affine[mask] = rotation.repeat(mask.sum(), 1, 1)
    cos_affine = torch.cos(v).view(-1, 1, 1) * torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32, device=x.device).reshape(-1, 2, 3)
    sin_affine = torch.sin(v).view(-1, 1, 1) * torch.tensor([0, 1, 0, -1, 0, 0], dtype=torch.float32, device=x.device).reshape(-1, 2, 3)
    affine = cos_affine + sin_affine

    x = apply_affine(x, affine)
    
    return x, affine.detach()


def Invert(x, v):
    affine = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32, device=x.device).reshape(-1, 2, 3).repeat(x.size(0), 1, 1)
    return 1.0 - x, affine