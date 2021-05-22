import torch

def create(config):
    if config['type'] == 'ce':
        return torch.nn.CrossEntropyLoss()
    else:
        raise AttributeError(f'not support loss config: {config}')
