import torch
import torchvision.models as models
import timm

from trainer.model.resnet import resnet32

def create(config):
    if config['type'] == 'R50x1':
        return models.resnet50(pretrained=False)

    # resnet32 implementation test
    elif config['type'] == 'resnet32':
        return resnet32()
    else:
        try:
            return timm.create_model(config['type'], pretrained=config.get('pretrained', False), num_classes=config['num_classes'])
        except:
            raise AttributeError(f'not support architecture config: {config}')