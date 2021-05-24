import torch
import torchvision

def create(config, model_params):

    if config['type'] == 'adamw':
        return torch.optim.AdamW(
            params=model_params,
            lr=config['lr']
        )
    elif config['type'] == 'sgd':
        return torch.optim.SGD(
            params = model_params,
            lr = config['lr'],
            weight_decay=config['weight_decay'],
            momentum=config['momentum']
        )
    else:
        raise AttributeError(f'not support optmizer config: {config}')
