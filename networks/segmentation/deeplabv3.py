from turtle import forward
from torchvision.models.segmentation.deeplabv3 import deeplabv3_resnet50
from torchvision.models.resnet import ResNet50_Weights

import torch
import torch.nn.functional as F
from collections import OrderedDict

class DeepLabEncapsulator(torch.nn.Module):
    def __init__(self, net) -> None:
        super().__init__()
        self.backbone = net.backbone
        self.classifier = net.classifier

    def forward(self, x):
        input_shape = x.shape[-2:]

        feats = self.backbone(x)["out"]
        scores = self.classifier(feats)
        
        scores = F.interpolate(scores, size=input_shape, mode="bilinear", align_corners=False)

        output = OrderedDict()
        output["out"] = scores
        output["feats"] = feats

        return output




def get_deeplab_v3(num_classes):
    return deeplabv3_resnet50(progress=True, num_classes=num_classes, aux_loss=False, weights_backbone=ResNet50_Weights.IMAGENET1K_V2)
