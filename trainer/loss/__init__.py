import torch

def create(config):
    if config['type'] == 'ce':
        
        losses = [torch.nn.CrossEntropyLoss(weight=w) for w in config['weight']]

        return losses
    else:
        raise AttributeError(f'not support loss config: {config}')
