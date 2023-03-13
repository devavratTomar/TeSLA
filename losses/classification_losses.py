import torch.nn as nn
import torch
import torch.nn.functional as F

class EntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, probs):
        # shape is [batch x n_classes]

        # returns un-reduced [batch] loss as well as  loss over ensembled prediction [batch]
        return torch.sum(-probs * torch.log(probs + 1e-8), dim=-1)


class EntropyClassMarginals(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, probs):
        # shape of probs is [batch x n_aug x n_classes]
        
        # calculate average prob across augmentations
        if probs.dim() == 3:
            probs = torch.mean(probs, dim=1) # [batch x n_classes]

        avg_p = probs.mean(dim=0) # avg along the batch [n_classes]
        entropy_cm = torch.sum(avg_p * torch.log(avg_p + 1e-8))
        return entropy_cm


class SCELossWithLogits(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes=10):
        super(SCELossWithLogits, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device) # batch x nclasses
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss

class SCELoss(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes=10):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.NLLLoss()

    def forward(self, pred, labels):
        # CCE pred are probs
        ce = self.cross_entropy(torch.log(pred + 1e-8), labels)

        # RCE
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device) # batch x nclasses
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss