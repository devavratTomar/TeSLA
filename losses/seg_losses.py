import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence

from torch import Tensor

def gan_loss(pred, should_be_classified_as_real):
    bs = pred.size(0)
    if should_be_classified_as_real:
        return F.softplus(-pred).view(bs, -1).mean(dim=1)
    else:
        return F.softplus(pred).view(bs, -1).mean(dim=1)


class SoftCrossEntropy(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, p_logits, q_probs):
        q_probs = q_probs.detach()
        loss = -(q_probs * torch.log_softmax(p_logits, dim=1)).sum(dim=1)
        return loss.mean()


class MumfordShahLoss(nn.Module):
    def __init__(self, param=0.001):
        super().__init__()
        self.param = param
        self.criterian_l2 = nn.MSELoss()
        self.criterian_l1 = nn.L1Loss()

    def calculate_tv(self, prob):
        G_x = prob[:, :, 1:, :] - prob[:, :, :-1, :]
        G_y = prob[:, :, :, 1:] - prob[:, :, :, :-1]

        G = torch.abs(G_x).sum() + torch.abs(G_y).sum()
        return G

    
    def calculate_level_set(self, target, output):
        ### softmax probability maps: output
        outshape = output.shape
        tarshape = target.shape
        loss = 0.0
        for ich in range(tarshape[1]):
            target_ = torch.unsqueeze(target[:,ich], 1)
            target_ = target_.expand(tarshape[0], outshape[1], tarshape[2], tarshape[3])
            pcentroid = torch.sum(target_ * output, (2,3))/(torch.sum(output, (2,3)) + 1e-6)
            pcentroid = pcentroid.view(tarshape[0], outshape[1], 1, 1)
            plevel = target_ - pcentroid.expand(tarshape[0], outshape[1], tarshape[2], tarshape[3])
            pLoss = plevel * plevel * output
            loss += torch.sum(pLoss)
        return loss


    def forward(self, img, prob):
        ## ignore background
        level_set_loss = self.calculate_level_set(img, prob)
        tv = self.calculate_tv(prob)

        loss = (level_set_loss + self.param * tv)/(img.shape[0])
        
        return loss

class EntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, probs):
        return torch.sum(-probs * torch.log(probs + 1e-8), dim=1).mean(dim=(1, 2))

class EntropyClassMarginals(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, probs):
        avg_p = probs.mean(dim=[0, 2, 3]) # avg along the pixels dim h x w -> size is batch x n_classes
        entropy_cm = torch.sum(avg_p * torch.log(avg_p + 1e-8), dim=-1)
        return entropy_cm

class CrossEntropyLossWeighted(nn.Module):
    """
    Cross entropy with instance-wise weights. Leave `aggregate` to None to obtain a loss
    vector of shape (batch_size,).
    """
    def __init__(self, n_classes=3):
        super(CrossEntropyLossWeighted, self).__init__()
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.n_classes = n_classes

    def one_hot(self, targets):
        targets_extend=targets.clone()
        targets_extend.unsqueeze_(1) # convert to Nx1xHxW
        one_hot = torch.FloatTensor(targets_extend.size(0), self.n_classes, targets_extend.size(2), targets_extend.size(3)).zero_().to(targets.device)
        one_hot.scatter_(1, targets_extend, 1)
        
        return one_hot
    
    def forward(self, inputs, targets):
        one_hot = self.one_hot(targets)

        # size is batch, nclasses, 256, 256
        weights = 1.0 - torch.sum(one_hot, dim=(2, 3), keepdim=True)/torch.sum(one_hot)
        one_hot = weights*one_hot

        loss = self.ce(inputs, targets).unsqueeze(1) # shape is batch, 1, 256, 256
        loss = loss*one_hot

        return torch.sum(loss)/(torch.sum(weights)*targets.size(0)*targets.size(1))


class ContourRegularizationLoss(nn.Module):
    def __init__(self, d=4):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2 * d + 1)

    def forward(self, x):
        # x is the probability maps
        C_d = self.max_pool(x) + self.max_pool(-1*x) # size is batch x 1 x h x w

        loss = torch.norm(C_d, p=2, dim=(2, 3)).mean()
        return loss


class NuclearNorm(nn.Module):
    def __init__(self, k=4):
        super().__init__()
        self.max_pool = nn.AvgPool2d(kernel_size=k)

    def forward(self, x):
        # x is probabilities
        x = self.max_pool(x) # size is batch x n_classes x h x w
        x = x.permute(0, 2, 3, 1) # batch x h x w x n_classes
        x = x.flatten(1, 2) # batch x hw x n_classes
        loss = torch.norm(x, "nuc", dim=(1, 2)).mean() / x.size(-1)
        return loss


class SCELoss(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device) # batch x h x w x nclasses
        
        label_one_hot = label_one_hot.permute(0, 3, 1, 2) if label_one_hot.dim() == 4 else label_one_hot # batch x nclasses x h x w
        
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce
        return loss


class FocalLoss(nn.Module):
    """ Focal Loss, as described in https://arxiv.org/abs/1708.02002.
    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.
    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(self,
                 alpha: Optional[Tensor] = None,
                 gamma: float = 2.,
                 reduction: str = 'mean',
                 ignore_index: int = -100):
        """Constructor.
        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction='none', ignore_index=ignore_index)

    def __repr__(self):
        arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return 0.
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


def focal_loss(alpha: Optional[Sequence] = None,
               gamma: float = 0.,
               reduction: str = 'mean',
               ignore_index: int = -100,
               device='cpu',
               dtype=torch.float32) -> FocalLoss:
    """Factory function for FocalLoss.
    Args:
        alpha (Sequence, optional): Weights for each class. Will be converted
            to a Tensor if not None. Defaults to None.
        gamma (float, optional): A constant, as described in the paper.
            Defaults to 0.
        reduction (str, optional): 'mean', 'sum' or 'none'.
            Defaults to 'mean'.
        ignore_index (int, optional): class label to ignore.
            Defaults to -100.
        device (str, optional): Device to move alpha to. Defaults to 'cpu'.
        dtype (torch.dtype, optional): dtype to cast alpha to.
            Defaults to torch.float32.
    Returns:
        A FocalLoss object
    """
    if alpha is not None:
        if not isinstance(alpha, Tensor):
            alpha = torch.tensor(alpha)
        alpha = alpha.to(device=device, dtype=dtype)

    fl = FocalLoss(
        alpha=alpha,
        gamma=gamma,
        reduction=reduction,
        ignore_index=ignore_index)
    return fl


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        target = target.clone()
        predict = F.softmax(predict, dim=1)
        target[target == self.ignore_index] = predict.shape[1]
        target_one_hot = F.one_hot(target, predict.shape[1] + 1).permute([0, 3, 1, 2]) # ch at dim 1
        for i in range(predict.shape[1]):
            dice_loss = dice(predict[:, i], target_one_hot[:, i])
            if self.weight is not None:
                assert self.weight.shape[0] == target_one_hot.shape[1], \
                    'Expect weight shape [{}], get[{}]'.format(target_one_hot.shape[1], self.weight.shape[0])
                dice_loss *= self.weights[i]
            total_loss += dice_loss

        return total_loss/predict.shape[1]