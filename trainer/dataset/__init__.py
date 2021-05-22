import torch
import torchvision
import torchvision.transforms as transforms
import os 

# dataloader sampler
def create(config, world_size=1, local_rank=-1, mode='train'):
    
    params = config[mode]

    if config['type'] == 'cifar10':
        transformers = transforms.Compose([preprocess(t) for t in params['preprocess']] )
        
        dataset = torchvision.datasets.CIFAR10(root='/root', download=True, transform=transformers)

        if local_rank >= 0:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=world_size, 
                rank=local_rank, 
                shuffle=params.get('shuffle', False)
            )
        else:
            sampler = None

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=params['batch_size'],
            num_workers=os.cpu_count(),
            pin_memory=True,
            drop_last=params.get('drop_last', False),
            sampler=sampler
        )

        return dataloader, sampler
    else:
        raise AttributeError(f'not support dataset config: {config}')

def preprocess(config):
    if config['type'] == 'pad':
        return transforms.Pad(**config['params'])
    elif config['type'] == 'randomcrop':
        return transforms.RandomCrop(**config['params'])
    elif config['type'] == 'horizontal':
        return transforms.RandomHorizontalFlip()
    elif config['type'] == 'tensor':
        return transforms.ToTensor()
    elif config['type'] == 'normalize':
        return transforms.Normalize(**config['params'])