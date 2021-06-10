import os
import sys
import logging

from tqdm import tqdm

import torch
import torch.cuda
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tensorboardX import SummaryWriter

import theconf
from theconf import Config as C

import numpy as np
from sklearn.metrics import recall_score

import trainer

from trainer.dataset.customdata import CustomDataset

summary = SummaryWriter()
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

    if C.get()['dataset']['type'] == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(root='/root', download=True, transform=transformers, train = params.get('train', False))
    elif C.get()['dataset']['type']== 'bengali':
        dataset = {}
        data = CustomDataset(C.get()['dataset']['train_img_path'], C.get()['dataset']['label_path'])
        dataset['train'], dataset['val'] = torch.utils.data.random_split(data, [160840, 40000], generator=torch.Generator().manual_seed(42))
    else:
        raise AttributeError(f'not support dataset config: {config}')

    train_loader, train_sampler = trainer.dataset.create(C.get()['dataset'],
                                              dataset, 
                                              int(os.environ.get('WORLD_SIZE', 1)), 
                                              int(os.environ.get('LOCAL_RANK', -1)),
                                              mode='train')

    val_loader, _ = trainer.dataset.create(C.get()['dataset'],
                                              dataset, 
                                              mode='val')
                
    # test_loader, _ = trainer.dataset.create(C.get()['dataset'],
    #                                           dataset,
    #                                           mode='test')

    optimizer = trainer.optimizer.create(C.get()['optimizer'], model.parameters())
    lr_scheduler = trainer.scheduler.create(C.get()['scheduler'], optimizer)

    criterion = trainer.loss.create(C.get()['loss']).to(device=device, non_blocking=True)

    if flags.local_rank >= 0:
        for name, x in model.state_dict().items():
            dist.broadcast(x, 0)
        torch.cuda.synchronize()

    is_best = 0
    for epoch in range(C.get()['scheduler']['epoch']):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        print(f'current lr {optimizer.param_groups[0]["lr"]:.5e}')
        lr_scheduler.step()
        train_acc, train_loss = train_one_epoch(epoch, model, train_loader, criterion, optimizer, device, flags)

        if epoch % 1 == 0 and flags.is_master:
            val_acc, val_loss = evaluate(epoch, model, val_loader, device, flags, criterion)
            is_best = val_acc if is_best < val_acc else is_best

        torch.cuda.synchronize()
        print(f'THE BEST MODEL val test :{is_best:3f}')
                
    ### tensorboard
        summary.add_scalar("Loss/train", train_loss, epoch)
        summary.add_scalar("Acc/train", train_acc, epoch)
        summary.add_scalar("Loss/val", val_loss, epoch)
        summary.add_scalar("Acc/val", val_acc, epoch)
        summary.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)
        summary.close()
        
def train_one_epoch(epoch, model, dataloader, criterion, optimizer, device, flags):
    one_epoch_loss = 0
    train_total = 0
    train_hit = 0

    model.train()

    if flags.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    scores = []
    for step, (image, label_a, label_b, label_c) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Train |{:3d}e".format(epoch), disable=not flags.is_master):
        image = image.to(device=device, non_blocking=True)
        label_a = label_a.to(device=device, non_blocking=True)
        label_b = label_b.to(device=device, non_blocking=True)
        label_c = label_c.to(device=device, non_blocking=True)

        if flags.use_amp:
            with torch.cuda.amp.autocast():
                y_pred = model(image)
                loss = criterion(y_pred[0], label_a) + criterion(y_pred[1], label_b) + criterion(y_pred[2], label_c)
                assert y_pred.dtype == torch.float16, f'{y_pred.dtype}'
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        else:
            y_pred = model(image)
            loss = criterion(y_pred[0], label_a) + criterion(y_pred[1], label_b) + criterion(y_pred[2], label_c)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        one_epoch_loss += loss.item()
        _, y_pred_a = y_pred[0].max(1)
        _, y_pred_b = y_pred[1].max(1)
        _, y_pred_c = y_pred[2].max(1)

        if flags.is_master:
            gather_y_true_a = [torch.ones_like(label_a) for _ in range(dist.get_world_size())]
            gather_y_true_b = [torch.ones_like(label_b) for _ in range(dist.get_world_size())]
            gather_y_true_c = [torch.ones_like(label_c) for _ in range(dist.get_world_size())]
            gather_y_pred_a = [torch.ones_like(y_pred_a) for _ in range(dist.get_world_size())]
            gather_y_pred_b = [torch.ones_like(y_pred_b) for _ in range(dist.get_world_size())]
            gather_y_pred_c = [torch.ones_like(y_pred_c) for _ in range(dist.get_world_size())]
            dist.all_gather(gather_y_true_a, label_a)
            dist.all_gather(gather_y_true_b, label_b)
            dist.all_gather(gather_y_true_c, label_c)
            dist.all_gather(gather_y_pred_a, y_pred_a)
            dist.all_gather(gather_y_pred_b, y_pred_b)
            dist.all_gather(gather_y_pred_c, y_pred_c)
            
            semi_scores = []
            semi_scores.append(recall_score(label_a.cpu().numpy(), y_pred_a.cpu().numpy(), average='macro'))
            semi_scores.append(recall_score(label_b.cpu().numpy(), y_pred_b.cpu().numpy(), average='macro'))
            semi_scores.append(recall_score(label_c.cpu().numpy(), y_pred_c.cpu().numpy(), average='macro'))

            ### DDP 하려면 이부분 수정 ###
            # semi_scores.append(recall_score(gather_y_true_a.detach().cpu().numpy(), gather_y_pred_a.detach().cpu().numpy(), average='macro'))
            # semi_scores.append(recall_score(gather_y_true_b, gather_y_pred_b, average='macro'))
            # semi_scores.append(recall_score(gather_y_true_c, gather_y_pred_c, average='macro'))
            scores.append(np.average(semi_scores, weights=[2,1,1]))
    
    if flags.is_master:
        final_score = np.average(scores)
        train_loss = one_epoch_loss / (step + 1)
        print( f'Epoch:{epoch + 1}' ,f'Losses: {train_loss}', f'macro-average-recall: {final_score}%')

    return final_score, train_loss

@torch.no_grad()
def evaluate(epoch, model, dataloader, device, flags, criterion):

    validation_losses = 0
    val_hit = 0
    val_total = 0

    model.eval()

    val_scores = []
    for step, (image, label_a, label_b, label_c) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Validation |{:3d}e".format(epoch), disable=not flags.is_master):
        image = image.to(device=device, non_blocking=True)
        label_a = label_a.to(device=device, non_blocking=True)
        label_b = label_b.to(device=device, non_blocking=True)
        label_c = label_c.to(device=device, non_blocking=True)

        y_pred = model(image)
        loss = criterion(y_pred[0], label_a) + criterion(y_pred[1], label_b) + criterion(y_pred[2], label_c)

        validation_losses += loss.item()
        _, y_pred_a = y_pred[0].max(1)
        _, y_pred_b = y_pred[1].max(1)
        _, y_pred_c = y_pred[2].max(1)
        
        val_semi_scores = []
        val_semi_scores.append(recall_score(label_a.cpu().numpy(), y_pred_a.cpu().numpy(), average='macro'))
        val_semi_scores.append(recall_score(label_b.cpu().numpy(), y_pred_b.cpu().numpy(), average='macro'))
        val_semi_scores.append(recall_score(label_c.cpu().numpy(), y_pred_c.cpu().numpy(), average='macro'))
        val_scores.append(np.average(val_semi_scores, weights=[2,1,1]))

    #     val_hit += torch.tensor(y_pred.clone().detach().eq(label).sum(), dtype=torch.int).to(device=device, non_blocking=True)
    #     val_total += torch.tensor(image.shape[0], dtype=torch.int).to(device=device, non_blocking=True)
    
    # val_acc = val_hit / val_total
    val_acc = np.average(val_scores)
    val_loss = validation_losses / (step + 1)
    print(f'Val Losses: {val_loss}', f'Val Acc: {val_acc * 100}%')
    return val_acc, val_loss

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