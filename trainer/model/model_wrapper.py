import timm
import torch.nn as nn


class model_wrapper(nn.Module):
    def __init__(self, configs):
        super(model_wrapper, self).__init__()

        self.model = timm.create_model(config['type'], pretrained=config.get('pretrained', False))
        self.features = self.model.fc.in_features
        self.model.fc = nn.Identity()
        self._fc = [nn.Linear(self.features, num_class) for num_class in config['num_classes']]


    def forward(self, x):
        output = self.model(x)
        outputs = [output for output in self._fc(output)]

        return output





