
import copy
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

class AdaptSupCEEfficientnet(nn.Module):
    def __init__(self, name='resnet50', num_classes=10):
        super(AdaptSupCEEfficientnet, self).__init__()
        self.encoder = models.__dict__[name](pretrained=True)
        self.fc = copy.deepcopy(self.encoder.classifier)
        self.encoder.classifier = nn.Identity()

        if num_classes != 1000:
            self.fc = nn.Sequential(*list(self.fc.children())[:-1], nn.Linear(list(self.fc.children())[-1].in_features, num_classes))

        for m in self.encoder.modules():
            if isinstance(m,nn.Dropout):
                m.p=0.0

        for m in self.fc.modules():
            if isinstance(m,nn.Dropout):
                m.p=0.0


    def forward(self, x, return_feats=False):
        feats = self.encoder(x)
        scores = F.softmax(self.fc(feats), dim=1)

        if return_feats:
            return scores, feats

        return scores