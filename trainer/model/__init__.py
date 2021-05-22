import torch
import torchvision.models as models

def create(config):
    if config['type'] == 'R50x1':
        return models.resnet50(pretrained=False)
    else:
        raise AttributeError(f'not support architecture config: {config}')
    