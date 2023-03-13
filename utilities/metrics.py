import torch

def dice_coef_bool(gt : torch.Tensor, preds : torch.Tensor):
    """Compute the dice coefficient between two boolean arrays
    Args:
        gt (torch.Tensor): The ground-trth arrays (a batch of array)
        preds (torch.Tensor): The prediction arrays (a batch of array)
    Returns:
        torch.Tensor: tensor array of dice score for each array in the batch
    """
    # get dims
    dims = tuple(torch.arange(gt.ndim-1)+1)

    # compute intersection score
    inter = torch.sum(gt * preds, dim=dims)

    # compute union
    union = (torch.sum(gt, dim=dims) + torch.sum(preds, dim=dims))

    # compute the dice score
    score = torch.zeros_like(inter, dtype=torch.float32)
    score[union>0] = 2*inter[union>0]/union[union>0]
    score[union==0] = torch.nan

    return score

def dice_coef_multiclass(gt : torch.Tensor, preds: torch.Tensor, n_classes):
    """In multiclass setting, computes the dice coefficient for each individual class
    Args:
        gt (torch.Tensor): The ground truth tensor (a batch of array)
        preds (torch.Tensor): The prediction tensor (a batch of array)
        classes: The list of classes to evaluate
    Returns:
        np.ndarray: an array with the dice coefficient for each class and each array in the batch
    """
    
    coefs = []
    for i in range(n_classes):
        coefs.append(dice_coef_bool(gt==i, preds==i))
    
    return torch.stack(coefs, dim=-1) # batch x n_classes



def iou_bool(gt : torch.Tensor, preds : torch.Tensor):
    """Compute the dice coefficient between two boolean arrays
    Args:
        gt (torch.Tensor): The ground-trth arrays (a batch of array)
        preds (torch.Tensor): The prediction arrays (a batch of array)
    Returns:
        torch.Tensor: tensor array of dice score for each array in the batch
    """
    # get dims
    dims = tuple(torch.arange(gt.ndim-1)+1)

    # compute intersection score
    inter = torch.sum(torch.bitwise_and(gt, preds), dim=dims)

    # compute union
    union = torch.sum(torch.bitwise_or(gt, preds), dim=dims)

    # compute the dice score
    score = torch.zeros_like(inter, dtype=torch.float32)
    score[union>0] = inter[union>0]/union[union>0]
    score[union==0] = torch.nan

    return score

def iou_multiclass(gt : torch.Tensor, preds: torch.Tensor, n_classes):
    """In multiclass setting, computes the dice coefficient for each individual class
    Args:
        gt (torch.Tensor): The ground truth tensor (a batch of array)
        preds (torch.Tensor): The prediction tensor (a batch of array)
        classes: The list of classes to evaluate
    Returns:
        np.ndarray: an array with the dice coefficient for each class and each array in the batch
    """
    
    coefs = []
    for i in range(n_classes):
        coefs.append(iou_bool(gt==i, preds==i))
    
    return torch.stack(coefs, dim=-1) # batch x n_classes