import torch
import torchvision.models as models

from trainer.model.resnet import resnet32

def create(config):
    if config['type'] == 'R50x1':
        return models.resnet50(pretrained=False)

    # resnet32 implementation test
    elif config['type'] == 'resnet32':
        return resnet32()

    else:
        raise AttributeError(f'not support architecture config: {config}')
    