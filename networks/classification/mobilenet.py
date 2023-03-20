
import copy
import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

class AdaptSupCEMobileNet(nn.Module):
    def __init__(self, name='mobilenet_v2', num_classes=10):
        super(AdaptSupCEMobileNet, self).__init__()
        self.encoder = models.__dict__[name](pretrained=True)
        self.fc = copy.deepcopy(self.encoder.classifier)
        self.encoder.classifier = nn.Identity()
        self.dim = list(self.fc.children())[-1].in_features

        if num_classes != 1000:
            self.fc = nn.Sequential(*list(self.fc.children())[:-1], nn.Linear(list(self.fc.children())[-1].in_features, num_classes))


    def forward(self, x, return_feats=False):
        feats = self.encoder(x)
        scores = self.fc(feats)

        if return_feats:
            return scores, feats

        return scores


def load_network_mobilenet(net, path):

    if path is not None:
        ckpt = torch.load(path, map_location="cpu")
        if "model" in ckpt.keys():
            state_dict = ckpt['model']
        else:
            state_dict = ckpt

        net_dict = {}

        for k, v in state_dict.items():

            k = k.replace("module.", "")
            k = k.replace("features.", "encoder.features.")
            k = k.replace("classifier.", "fc.")
            net_dict[k] = v

        net.load_state_dict(net_dict, strict=True)