import sys
import os
import time
import numpy as np
import random
import torch
import utils
from hhutil.io import time_now

import torch.nn.functional as F
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
from torchsummary import summary

from models.cifar.preact_resnet import WRN_28_10
from models.cifar.shakeshake import ShakeResNet
from dataset import CIFAR_MEAN, CIFAR_STD
from dataset.load_data import load_cifar_data
from lr_scheduler import CosineLR
from loss import DRASearchLoss
from utils import accuracy

# Params to replace
task_id = 'TASK_ID'
save_path = 'CKPT_PATH'
data_root = 'CIFAR_ROOT'

# Save settings
save_ckpt_freq = 5
continue_training = True

# Device settings
gpu_device_id = 0
torch.cuda.set_device(gpu_device_id)

# Training settings
use_seed = False
seed = 0
grad_clip = None
use_fp16 = True
label_smoothing_b1 = 0.0
label_smoothing_b2 = 0.0

# for WRN
batch_size = 128 * 2
eval_batch_size = 1024
base_lr = 0.1
epochs = 200
warmup_epochs = 5
wd = 5e-4

# # for ShakeShake
# batch_size = 128 * 2
# eval_batch_size = 1024
# base_lr = 0.01
# epochs = 1800
# warmup_epochs = 5
# wd = 1e-3

# DRA settings
augment_space = 'RA'
aug_depth = 2
p_min_t = 1.0
p_max_t = 1.0

# Basic augmentation settings
cutout_len = 16

# Model settings
model_name = 'WRN_28_10'
#model_name = 'ShakeShake_26_2x96d'
dropout = 0.0

# Dataset settings
DATASET = 'CIFAR10'
#DATASET = 'CIFAR100'

if DATASET == 'CIFAR10':
    NUM_CLASS = 10
    RES = 32
elif DATASET == 'CIFAR100':
    NUM_CLASS = 100
    RES = 32


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
    train_transform = transforms.Compose([
        transforms.RandomCrop(RES, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
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
    if model_name == 'WRN_28_10':
        model = WRN_28_10(num_classes=NUM_CLASS, dropout=dropout,
                          aug_depth=aug_depth,
                          resolution=RES, augment_space=augment_space, p_min_t=p_min_t, p_max_t=p_max_t,
                          cutout_len=cutout_len,
                          norm_mean=CIFAR_MEAN, norm_std=CIFAR_STD).cuda()
    elif model_name == 'ShakeShake_26_2x96d':
        model = ShakeResNet(depth=26, w_base=96, label=NUM_CLASS,
                         aug_depth=aug_depth,
                         resolution=RES, augment_space=augment_space, p_min_t=p_min_t, p_max_t=p_max_t,
                         cutout_len=cutout_len,
                         norm_mean=CIFAR_MEAN, norm_std=CIFAR_STD).cuda()
    else:
        raise NotImplementedError('Model to implement!')
    summary(model, (3, RES, RES), device='cuda')

    train_trans, test_trans = basic_transforms()
    train_loader, _, test_loader = load_cifar_data(DATASET, batch_size, eval_batch_size,
                                                   root=data_root, train_transform=train_trans,
                                                   test_transform=test_trans,
                                                   use_proxy=False, num_proxy=None)
    model_optimizer = torch.optim.SGD(model.parameters(), lr=base_lr,
                                      momentum=0.9, weight_decay=wd, nesterov=True)
    steps_per_epoch = len(train_loader)
    model_scheduler = CosineLR(model_optimizer, steps_per_epoch * epochs * 2, min_lr=0,
                               warmup_epoch=steps_per_epoch * warmup_epochs * 2, warmup_min_lr=0)   # *2: One iteration with 2 updates

    start_epoch = 0
    if continue_training and os.path.exists(save_path):
        ckpt = torch.load(save_path, map_location=lambda storage, loc: storage.cuda(gpu_device_id))
        model.load_state_dict(ckpt['model_state_dict'])
        model_optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        model_scheduler.last_epoch += start_epoch * steps_per_epoch * 2
        print('Start training from ckpt %s.' % save_path)

    for epoch in range(start_epoch, epochs):
        print('%s Epoch %d/%d' % (time_now(), epoch + 1, epochs))

        # training
        train_loss, train_acc = train_epoch(train_loader, model, model_optimizer, model_scheduler, criterion)
        print('%s [Train] loss: %.4f, acc: %.4f' % (time_now(), train_loss, train_acc))

        if save_ckpt_freq is not None and (epoch + 1) % save_ckpt_freq == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'optimizer_state_dict': model_optimizer.state_dict(),
            }, save_path)

        # inference
        val_loss, val_acc = infer(test_loader, model, criterion)
        print('%s [Valid] loss: %.4f, acc: %.4f' % (time_now(), val_loss, val_acc))

    print('%s End training' % time_now())
    end_time = time.time()
    elapsed = end_time - start_time
    print('Search time: %.3f Hours' % (elapsed / 3600.))


def train_epoch(train_loader, model, train_optim, model_scheduler, criterion):
    train_loss_m = utils.AverageMeter()
    train_top1 = utils.AverageMeter()
    scaler = GradScaler(enabled=use_fp16)

    for step, train_data in enumerate(train_loader):
        model.train()

        train_input, train_label = [x.cuda() for x in train_data]
        N = train_input.shape[0]

        # split two batches
        train_input_b1 = train_input[:N // 2]
        train_input_b2 = train_input[N // 2:]
        train_label_b1 = train_label[:N // 2]
        train_label_b2 = train_label[N // 2:]

        # First: update weights with random augmentation
        cos_sim_1 = torch.zeros((aug_depth, N // 2)).uniform_(0, 1)
        train_optim.zero_grad()
        with autocast(enabled=use_fp16):
            train_pred_b1 = model(train_input_b1, training=True, cos=cos_sim_1, use_basic_aug=False)
            train_loss = criterion(train_pred_b1, train_label_b1, search=True, lb_smooth=label_smoothing_b1)
        scaler.scale(train_loss).backward()
        if use_fp16:
            scaler.unscale_(train_optim)
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(train_optim)
        scaler.update()
        model_scheduler.step()

        train_prec1 = accuracy(train_pred_b1[0].detach(), train_label_b1.detach(), topk=(1,))
        train_loss_m.update(train_loss.detach().item(), N // 2)
        train_top1.update(train_prec1.detach().item(), N // 2)

        # Second: calculate prediction of the second batch
        with torch.no_grad():
            train_pred_ori_b2 = model(train_input_b2, training=True, y=train_label_b2, use_basic_aug=False)
        # pred_ori_2 = train_pred_ori_2[1]
        cos_sim_2 = train_pred_ori_b2[2]   # use similarity augment
        image_basic = train_pred_ori_b2[3]

        # Third: update weights with augmented images
        train_optim.zero_grad()
        with autocast(enabled=use_fp16):
            train_pred_b2 = model(image_basic, training=True, cos=cos_sim_2, use_basic_aug=False)
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
        train_prec1 = accuracy(train_pred_b2[0].detach(), train_label_b2.detach(), topk=(1,))
        train_loss_m.update(train_loss.detach().item(), N - N // 2)
        train_top1.update(train_prec1.detach().item(), N - N // 2)

    return train_loss_m.avg, train_top1.avg


def infer(test_loader, model, criterion):
    loss_m = utils.AverageMeter()
    top1 = utils.AverageMeter()

    model.eval()
    with torch.no_grad():
        for step, data in enumerate(test_loader):
            input, label = [x.cuda() for x in data]
            N = input.shape[0]

            pred = model(input, training=False)
            loss = criterion(pred, label, search=False)
            prec1 = accuracy(pred[0], label, topk=(1,))

            loss_m.update(loss.detach().item(), N)
            top1.update(prec1.detach().item(), N)

    return loss_m.avg, top1.avg


if __name__ == '__main__':
    main()