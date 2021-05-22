import os
import sys
import logging

from tqdm import tqdm

import torch
import torch.cuda
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


import theconf
from theconf import Config as C

import trainer

def main(flags):
    device = torch.device(type= 'cuda', index=max(0, int(os.environ.get('LOCAL_RANK', -1))))
    if flags.local_rank >= 0:
        dist.init_process_group(backend=flags.dist_backend, init_method= 'env://', world_size=int(os.environ['WORLD_SIZE']))
        torch.cuda.set_device(device)

        flags.is_master = flags.local_rank < 0 or dist.get_rank() == 0

        C.get()['optimizer']['lr'] *= dist.get_world_size()
        flags.optimizer_lr = C.get()['optimizer']['lr']
        if flags.is_master:
            print(f"local batch={C.get()['dataset']['train']['batch_size']}, world_size={dist.get_world_size()} ----> total batch={C.get()['dataset']['train']['batch_size'] * dist.get_world_size()}")
            print(f"lr -> {C.get()['optimizer']['lr']}")

    torch.backends.cudnn.benchmark = True

    model = trainer.model.create(C.get()['architecture'])
    model.to(device=device, non_blocking=True)

    if flags.local_rank >= 0:
        model = DDP(model, device_ids=[flags.local_rank], output_device=flags.local_rank)

    train_loader, train_sampler = trainer.dataset.create(C.get()['dataset'],
                                              int(os.environ.get('WORLD_SIZE', 1)), 
                                              int(os.environ.get('LOCAL_RANK', -1)),
                                              mode='train')
    test_loader, _ = trainer.dataset.create(C.get()['dataset'],
                                              mode='test')
    optimizer = trainer.optimizer.create(C.get()['optimizer'], model.parameters())
    lr_scheduler = trainer.scheduler.create(C.get()['scheduler'], optimizer)

    criterion = trainer.loss.create(C.get()['loss']).to(device=device, non_blocking=True)

    if flags.local_rank >= 0:
        for name, x in model.state_dict().items():
            dist.broadcast(x, 0)
        torch.cuda.synchronize()


    for epoch in range(C.get()['scheduler']['epoch']):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        lr_scheduler.step()
        train_one_epoch(epoch, model, train_loader, criterion, optimizer, device, flags)

        if epoch % 10 == 0 and flags.is_master:
            evaluate(epoch, model, test_loader, device, flags)
        
    

    
def train_one_epoch(epoch, model, dataloader, criterion, optimizer, device, flags):
    one_epoch_loss = 0
    train_total = 0
    train_hit = 0

    if flags.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    for step, (image, label) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Train |{:3d}e".format(epoch), disable=not flags.is_master):
        image = image.to(device=device, non_blocking=True)
        label = label.to(device=device, non_blocking=True)

        
        if flags.use_amp:
            with torch.cuda.amp.autocast():
                y_pred = model(image)
                loss = criterion(y_pred, label)
                assert y_pred.dtype == torch.float16, f'{y_pred.dtype}'
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        else:
            y_pred = model(image)
            loss = criterion(y_pred, label)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        

        one_epoch_loss += loss.item()
        _, y_pred = y_pred.max(1)            
        # train_hit += torch.tensor(y_pred.clone().detach().eq(label).sum(), dtype=torch.int).to(device=device, non_blocking=True)
        # train_total += torch.tensor(image.shape[0], dtype=torch.int).to(device=device, non_blocking=True)

        
    if flags.is_master:
    #     gather_hit = [torch.tensor([0], dtype=torch.float).to(device=device) for _ in range(dist.get_world_size())] 
    #     torch.distributed.all_gather(gather_hit, train_hit)

    #     gather_total = [torch.tensor([0], dtype=torch.float).to(device=device) for _ in range(dist.get_world_size())] 
    #     torch.distributed.all_gather(gather_total, train_total)

    #     print(gather_hit)
    #     print(gather_total)
    #     train_acc = sum(gather_hit) / sum(gather_total)
        print(f'Losses: {one_epoch_loss / (step + 1)}')
    #     print(f'Acc: {train_acc * 100}%')
    #     print(f'Train total: {gather_total}')

@torch.no_grad()
def evaluate(epoch, model, dataloader, device, flags):

    validation_loss = 0

    for step, (image, label) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Validation |{:3d}e".format(epoch), disable=not flags.is_master):
        image = image.to(device=device, non_blocking=True)
        label = label.to(device=device, non_blocking=True)

        y_pred = model(image)






if __name__ == '__main__':
    parser = theconf.ConfigArgumentParser(conflict_handler='resolve')
    parser.add_argument('--seed', type=lambda x: int(x, 0), default=None, help='set seed (default:0xC0FFEE)')

    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='Used for multi-process training. Can either be manually set ' +
                             'or automatically set by using \'python -m torch.distributed.launch\'.')
    
    parser.add_argument('--use_amp', action='store_true')

    flags = parser.parse_args()
    

    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s', stream=sys.stderr)

    main(flags)