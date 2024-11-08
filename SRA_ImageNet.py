import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import time
import numpy as np
import random
import torch
import utils
from PIL import Image
from hhutil.io import time_now

import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms as trans
from torchsummary import summary

from models.imagenet.resnet_timm import resnet50, resnet200
from models.imagenet.deit import DeiT
from models.imagenet.swin import Swin
from models.imagenet.vmamba import VMamba
from dataset import IMAGENET_STD, IMAGENET_MEAN
from dataset.load_data_distributed import load_imagenet_data
from lr_scheduler import CosineLR
from loss import DRASearchLoss
from utils import accuracy



# Params to replace
task_id = 'TASK_ID'
save_path = 'CKPT_PATH'
data_root = 'IMAGENET_ROOT'

# Device settings
local_rank = int(os.environ.get("LOCAL_RANK", -1))
use_multi_gpu = True
if use_multi_gpu:
    dist.init_process_group(backend="nccl")

# Save settings
save_ckpt_freq = 1
continue_training = True

# Training settings
use_seed = False
seed = 0
grad_clip = None
use_fp16 = True
label_smoothing_b1 = 0.1
label_smoothing_b2 = 0.1

# # for Res-50/200
# batch_size = 1024 * 2        # total batchsize, not for per GPU
# num_gpu = 4
# eval_batch_size = 1024
# base_lr = 0.4
# warmup_epochs = 5
# epochs = 270
# wd = 1e-4
# droppath = None
# dropout = 0.0

# # for DeiT
# batch_size = 1024 * 2        # total batchsize, not for per GPU
# num_gpu = 4
# eval_batch_size = 1024
# base_lr = 1e-3
# warmup_epochs = 5
# epochs = 300
# wd = 0.05
# droppath = 0.1
# dropout = 0.0

# # for Swin
# batch_size = 1024 * 2        # total batchsize, not for per GPU
# num_gpu = 4
# eval_batch_size = 1024
# base_lr = 1e-3
# warmup_epochs = 20
# epochs = 300
# wd = 0.05
# droppath = 0.1
# dropout = 0.0

# for VMamba
batch_size = 1024 * 2        # total batchsize, not for per GPU
num_gpu = 4
eval_batch_size = 1024
base_lr = 1e-3
warmup_epochs = 20
epochs = 300
wd = 0.05
droppath = 0.2
dropout = None

# DRA settings
augment_space = 'RA'
scale = 1.
aug_depth = 2
p_min_t = 1.0
p_max_t = 1.0

# Basic Augmentation settings
res_train = 224
res_val_resize = 256
res_val = 224

# Model settings
# model_name = 'ResNet-50'
#model_name = 'ResNet-200'
# model_name = 'deit_tiny_patch16_224'
# model_name = 'swin_tiny_patch4_window7_224'
model_name = 'vmamba_tiny'


# Dataset settings
NUM_CLASS = 1000


def reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.autograd.set_detect_anomaly(True)


def basic_transforms():
    train_transform = trans.Compose([
        trans.RandomResizedCrop(res_train, scale=(0.08, 1.0), interpolation=trans.InterpolationMode.BICUBIC),
        trans.RandomHorizontalFlip(),
        # trans.ColorJitter(
        #     brightness=0.4,
        #     contrast=0.4,
        #     saturation=0.4,
        # ),
        trans.ToTensor(),
    ])

    test_transform = trans.Compose([
        trans.Resize(res_val_resize),
        trans.CenterCrop(res_val),
        trans.ToTensor(),
    ])
    return train_transform, test_transform


def main():
    start_time = time.time()
    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)

    if use_seed:
        reproducibility(seed=seed)

    criterion = DRASearchLoss().cuda()
    if model_name == 'ResNet-50':
        model = resnet50(num_classes=NUM_CLASS, dropout=dropout,
                         aug_depth=aug_depth,
                         resolution=res_train, augment_space=augment_space, p_min_t=p_min_t, p_max_t=p_max_t,
                         norm_mean=IMAGENET_MEAN, norm_std=IMAGENET_STD)
        model_optimizer = torch.optim.SGD(model.parameters(), lr=base_lr,
                                          momentum=0.9, weight_decay=wd, nesterov=True)
    elif model_name == 'ResNet-200':
        model = resnet200(num_classes=NUM_CLASS, dropout=dropout,
                          aug_depth=aug_depth,
                          resolution=res_train, augment_space=augment_space, p_min_t=p_min_t, p_max_t=p_max_t,
                          norm_mean=IMAGENET_MEAN, norm_std=IMAGENET_STD)
        model_optimizer = torch.optim.SGD(model.parameters(), lr=base_lr,
                                          momentum=0.9, weight_decay=wd, nesterov=True)
    elif model_name == 'deit_tiny_patch16_224':
        model = DeiT(model_name=model_name,
                     num_classes=NUM_CLASS, dropout=dropout, droppath=droppath,
                     aug_depth=aug_depth,
                     resolution=res_train, augment_space=augment_space, p_min_t=p_min_t, p_max_t=p_max_t,
                     norm_mean=IMAGENET_MEAN, norm_std=IMAGENET_STD)
        model_optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=wd)
    elif model_name == 'swin_tiny_patch4_window7_224':
        model = Swin(model_name=model_name,
                     num_classes=NUM_CLASS, dropout=dropout, droppath=droppath,
                     aug_depth=aug_depth,
                     resolution=res_train, augment_space=augment_space, p_min_t=p_min_t, p_max_t=p_max_t,
                     norm_mean=IMAGENET_MEAN, norm_std=IMAGENET_STD)
        model_optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=wd)
    elif model_name == 'vmamba_tiny':
        model = VMamba(num_classes=NUM_CLASS, droppath=droppath,
                     aug_depth=aug_depth,
                     resolution=res_train, augment_space=augment_space, p_min_t=p_min_t, p_max_t=p_max_t,
                     norm_mean=IMAGENET_MEAN, norm_std=IMAGENET_STD)
        model_optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=wd)
    else:
        raise NotImplementedError('Model to implement!')

    train_trans, test_trans = basic_transforms()
    train_loader, _, test_loader = load_imagenet_data(batch_size // num_gpu, eval_batch_size, root=data_root,
            train_transform=train_trans, test_transform=test_trans, use_proxy=False, use_multi_gpu=use_multi_gpu)
    steps_per_epoch = len(train_loader)
    model_scheduler = CosineLR(model_optimizer, steps_per_epoch * epochs * 2, min_lr=0,
                               warmup_epoch=steps_per_epoch * warmup_epochs * 2, warmup_min_lr=0)

    if continue_training and save_path is not None and os.path.exists(save_path):
        if use_multi_gpu:
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        ckpt = torch.load(save_path, map_location=device)
        
        weights_dict = {}
        for k, v in ckpt['model_state_dict'].items():
            new_k = k.replace('module.', '') if 'module' in k else k
            weights_dict[new_k] = v
        ckpt['model_state_dict'] = weights_dict
        
        model.load_state_dict(ckpt['model_state_dict'])
        model_optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        model_scheduler.last_epoch += start_epoch * steps_per_epoch * 2
        if use_multi_gpu:
            #device = torch.device(f"cuda:{local_rank}")
            model = torch.nn.parallel.DistributedDataParallel(model.to(device),
                                                              device_ids=[local_rank],
                                                              output_device=local_rank)
            for state in model_optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)
        else:
            #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            for state in model_optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)
        print('Start training from ckpt %s.' % save_path)
    else:
        start_epoch = 0
        if use_multi_gpu:
            device = torch.device(f"cuda:{local_rank}")
            model = torch.nn.parallel.DistributedDataParallel(
                model.to(device),
                device_ids=[local_rank],
                output_device=local_rank
            )
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

    for epoch in range(start_epoch, epochs):
        if local_rank == 0:
            print('%s Epoch %d/%d' % (time_now(), epoch + 1, epochs))
        train_loader.sampler.set_epoch(epoch)

        # training
        train_loss, train_acc1, train_acc5 = train_epoch(train_loader, model, model_optimizer, model_scheduler, criterion, device)
        if local_rank == 0:
            print('%s [Train] loss: %.4f, top1_acc: %.4f, top5_acc: %.4f' % (time_now(), train_loss, train_acc1, train_acc5))

        if save_ckpt_freq is not None and (epoch + 1) % save_ckpt_freq == 0 and save_path is not None:
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'optimizer_state_dict': model_optimizer.state_dict(),
            }, save_path)
            if local_rank == 0:
                print('Save parameters to %s' % (save_path))

        # inference
        if local_rank == 0:
            val_loss, val_acc1, val_acc5 = infer(test_loader, model, criterion, device)
            print('%s [Valid] loss: %.4f, top1_acc: %.4f, top5_acc: %.4f' % (time_now(), val_loss, val_acc1, val_acc5))
        #torch.distributed.barrier()

    if local_rank == 0:
        print('%s End training' % time_now())
    end_time = time.time()
    elapsed = end_time - start_time
    if local_rank == 0:
        print('Training time: %.3f Hours' % (elapsed / 3600.))


def train_epoch(train_loader, model, train_optim, model_scheduler, criterion, device):
    train_loss_m = utils.AverageMeter()
    train_top1 = utils.AverageMeter()
    train_top5 = utils.AverageMeter()
    scaler = GradScaler(enabled=use_fp16)

    for step, train_data in enumerate(train_loader):
        model.train()

        train_input, train_label = [x.to(device) for x in train_data]
        N = train_input.shape[0]
        # split two batches
        train_input_b1 = train_input[:N // 2]
        train_input_b2 = train_input[N // 2:]
        train_label_b1 = train_label[:N // 2]
        train_label_b2 = train_label[N // 2:]

        # First: update weights with original images
        cos_sim_1 = torch.zeros((aug_depth, N // 2)).uniform_(0, 1)
        train_optim.zero_grad()
        with autocast(enabled=use_fp16):
            train_pred_ori = model(train_input_b1, training=True, cos=cos_sim_1)
            train_loss = criterion(train_pred_ori, train_label_b1, search=True, lb_smooth=label_smoothing_b1)
        scaler.scale(train_loss).backward()
        if use_fp16:
            scaler.unscale_(train_optim)
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(train_optim)
        scaler.update()
        model_scheduler.step()

        # Update metrics
        train_prec1, train_prec5 = accuracy(train_pred_ori[0].detach(), train_label_b1.detach(), topk=(1, 5))
        train_loss_m.update(train_loss.detach().item(), N // 2)
        train_top1.update(train_prec1.detach().item(), N // 2)
        train_top5.update(train_prec5.detach().item(), N // 2)

        # Second: calculate prediction of the second batch
        with torch.no_grad():
            train_pred_ori_b2 = model(train_input_b2, training=True, y=train_label_b2)
        cos_sim_2 = train_pred_ori_b2[2]   # use similarity augment

        # Third: update weights with augmented images
        train_optim.zero_grad()
        with autocast(enabled=use_fp16):
            train_pred_b2 = model(train_input_b2, training=True, cos=cos_sim_2)
            train_loss = criterion(train_pred_b2, train_label_b2, search=True, lb_smooth=label_smoothing_b2)
        scaler.scale(train_loss).backward()
        if use_fp16:
            scaler.unscale_(train_optim)
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(train_optim)
        scaler.update()
        model_scheduler.step()

        # Update metrics
        train_prec1, train_prec5 = accuracy(train_pred_b2[0].detach(), train_label_b2.detach(), topk=(1, 5))
        train_loss_m.update(train_loss.detach().item(), N - N // 2)
        train_top1.update(train_prec1.detach().item(), N - N // 2)
        train_top5.update(train_prec5.detach().item(), N - N // 2)

    return train_loss_m.avg, train_top1.avg, train_top5.avg


def infer(test_loader, model, criterion, device):
    loss_m = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    model.eval()
    with torch.no_grad():
        for step, data in enumerate(test_loader):
            input, label = [x.to(device) for x in data]
            N = input.shape[0]

            pred = model(input, training=False)
            loss = criterion(pred, label, search=False)
            prec1, prec5 = accuracy(pred[0], label, topk=(1, 5))

            loss_m.update(loss.detach().item(), N)
            top1.update(prec1.detach().item(), N)
            top5.update(prec5.detach().item(), N)

    return loss_m.avg, top1.avg, top5.avg


if __name__ == '__main__':
    main()