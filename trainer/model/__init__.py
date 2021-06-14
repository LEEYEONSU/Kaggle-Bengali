import torch
import torchvision.models as models
import timm

from .resnet import resnet32
from .model_wrapper import model_wrapper

def create(configs):
    if configs['type'] == 'R50x1':
        return models.resnet50(pretrained=False)
    # resnet32 implementation test
    elif configs['type'] == 'resnet32':
        return resnet32()
    else:
        try:
            return model_wrapper(configs)
        except:
            raise AttributeError(f'not support architecture config: {configs}')