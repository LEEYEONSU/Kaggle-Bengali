import torch

def create(config, optimizer):
    if config['type'] == 'multistep':
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, **config['params'])

    else:
        raise AttributeError(f'not support scheduler config: {config}')
