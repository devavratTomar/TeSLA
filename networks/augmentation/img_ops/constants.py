import torch

def gaussian_window(window_size):
    def gauss_fcn(x):
        return -(x - window_size // 2)**2 / 2.0
    
    gauss = torch.stack(
        [torch.exp(torch.tensor(gauss_fcn(x))) for x in range(window_size)])
    return gauss

def get_gaussian_kernel(ksize):
    window_1d = gaussian_window(ksize)
    return window_1d

def get_gaussian_kernel2d(ksize):
    kernel_x = get_gaussian_kernel(ksize)
    kernel_y = get_gaussian_kernel(ksize)
    kernel_2d = torch.matmul(
        kernel_x.unsqueeze(-1), kernel_y.unsqueeze(-1).t())
    return kernel_2d

GAUSSIAN_KERNEL = get_gaussian_kernel2d(7).repeat(1, 1, 1, 1)

SMOOTH_KERNEL   = torch.tensor([ [1, 1, 1], [1, 5, 1],  [1, 1, 1]], dtype=torch.float32).repeat(1, 1, 1, 1) / 13.0