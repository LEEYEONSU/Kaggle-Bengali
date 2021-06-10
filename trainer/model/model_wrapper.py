import timm
import torch
import torch.nn as nn
import numpy as np

class model_wrapper(nn.Module):
    def __init__(self, configs):
        super(model_wrapper, self).__init__()

        self.model = timm.create_model(configs['type'], pretrained=configs.get('pretrained', False))
        self.features = self.model.fc.in_features
        self.model.fc = nn.Identity()
        self._fc = nn.ModuleList([nn.Linear(self.features, num_class) for num_class in configs['num_classes']])

    def forward(self, x):
        output = self.model(x)
        outputs = [m(output) for m in self._fc]

        return outputs







