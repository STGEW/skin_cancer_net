import torch
import torch.nn as nn
from torchvision.models import inception_v3
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, out_channels):
        super(Model, self).__init__()
        
        self.model = inception_v3(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, out_channels)

    def forward(self, images):
        features = self.model(images)

        return features
