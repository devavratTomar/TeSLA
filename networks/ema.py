
import torch
import torch.nn as nn

from copy import deepcopy
from torch.nn.utils.weight_norm import WeightNorm


class ModelEMA(nn.Module):

    def __init__(self, model, decay=0.9, updates=0):
        # Create EMA
        super().__init__()
        self.updates = updates  # number of EMA updates
        self.decay = decay
        self.orig_decay = decay
        self.reset(model)


    def __call__(self, *args, **kwargs):
        output = self.ema(*args, **kwargs)
        return output
    

    def reset(self, model):
        weight_norm=None
        for module in model.modules():
            for _, hook in module._forward_pre_hooks.items():
                if isinstance(hook, WeightNorm):
                    weight_norm = hook
                    delattr(module, hook.name)

        self.ema = deepcopy(model)

        for module in model.modules():
            for _, hook in module._forward_pre_hooks.items():
                if isinstance(hook, WeightNorm):
                    hook(module, None)

        if weight_norm is not None:
            self.ema.fc.register_forward_pre_hook(weight_norm)

        for p in self.ema.parameters():
            p.requires_grad_(False)


    def lr_scheduler(self, iter_num, max_iter, gamma=10, power=0.75):
        decay = (1 + gamma * iter_num / max_iter) ** (-power)
        self.decay = 1. - (decay*(1.-self.orig_decay))


    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            d = self.decay

            for param_q, param_k in zip(model.parameters(), self.ema.parameters()):
                param_k.data.mul_(d).add_((1 - d) * param_q.detach().data)
