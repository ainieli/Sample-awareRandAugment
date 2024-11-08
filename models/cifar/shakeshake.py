import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from transforms.SRA import SampleAwareRandAugment as SRA
from transforms import Cutout, Normalize
from transforms.operations import HFlip, PadCrop

class ShakeShake(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x1, x2, training=True):
        if training:
            alpha = torch.FloatTensor(x1.size(0)).uniform_().to(x1.device)
            alpha = alpha.view(alpha.size(0), 1, 1, 1).expand_as(x1)
        else:
            alpha = 0.5
        return alpha * x1 + (1 - alpha) * x2

    @staticmethod
    def backward(ctx, grad_output):
        beta = torch.FloatTensor(grad_output.size(0)).uniform_().to(grad_output.device)
        beta = beta.view(beta.size(0), 1, 1, 1).expand_as(grad_output)
        beta = Variable(beta)

        return beta * grad_output, (1 - beta) * grad_output, None


class Shortcut(nn.Module):

    def __init__(self, in_ch, out_ch, stride):
        super(Shortcut, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(in_ch, out_ch // 2, 1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(in_ch, out_ch // 2, 1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        h = F.relu(x)

        h1 = F.avg_pool2d(h, 1, self.stride)
        h1 = self.conv1(h1)

        h2 = F.avg_pool2d(F.pad(h, (-1, 1, -1, 1)), 1, self.stride)
        h2 = self.conv2(h2)

        h = torch.cat((h1, h2), 1)
        return self.bn(h)


class ShakeBlock(nn.Module):

    def __init__(self, in_ch, out_ch, stride=1):
        super(ShakeBlock, self).__init__()
        self.equal_io = in_ch == out_ch
        self.shortcut = self.equal_io and None or Shortcut(in_ch, out_ch, stride=stride)

        self.branch1 = self._make_branch(in_ch, out_ch, stride)
        self.branch2 = self._make_branch(in_ch, out_ch, stride)

    def forward(self, x):
        h1 = self.branch1(x)
        h2 = self.branch2(x)
        h = ShakeShake.apply(h1, h2, self.training)
        h0 = x if self.equal_io else self.shortcut(x)
        return h + h0

    def _make_branch(self, in_ch, out_ch, stride=1):
        return nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_ch))


class ShakeResNet(nn.Module):

    def __init__(self, depth, w_base, label,
                 aug_depth=2,
                 resolution=32, augment_space='RA', p_min_t=0.2, p_max_t=0.8,
                 cutout_len=16,
                 norm_mean=[0.491, 0.482, 0.447], norm_std=[0.247, 0.243, 0.262]
                 ):
        super(ShakeResNet, self).__init__()
        n_units = (depth - 2) / 6

        in_chs = [16, w_base, w_base * 2, w_base * 4]
        self.in_chs = in_chs

        self.c_in = nn.Conv2d(3, in_chs[0], kernel_size=3, stride=1, padding=1)
        self.layer1 = self._make_layer(n_units, in_chs[0], in_chs[1])
        self.layer2 = self._make_layer(n_units, in_chs[1], in_chs[2], 2)
        self.layer3 = self._make_layer(n_units, in_chs[2], in_chs[3], 2)
        self.fc_out = nn.Linear(in_chs[3], label)

        self.sra = SRA(depth=aug_depth,
                   resolution=resolution, augment_space=augment_space,
                   p_min_t=p_min_t, p_max_t=p_max_t
                   )
        self.pad_crop = PadCrop(size=resolution, padding=4)
        self.hflip = HFlip()
        self.cutout = Cutout(cutout_len)
        self.normalize = Normalize(norm_mean, norm_std)

        # Initialize paramters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
                torch.nn.init.zeros_(m.bias)

    def net_forward(self, x, training=True):
        h = self.c_in(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = F.relu(h)
        h = F.avg_pool2d(h, 8)
        h = h.view(-1, self.in_chs[3])
        if training:
            features = h
            h = self.fc_out(h)
            return h, features
        else:
            h = self.fc_out(h)
            return h

    def _make_layer(self, n_units, in_ch, out_ch, stride=1):
        layers = []
        for i in range(int(n_units)):
            layers.append(ShakeBlock(in_ch, out_ch, stride=stride))
            in_ch, stride = out_ch, 1
        return nn.Sequential(*layers)

    def forward(self, x, training=False, y=None, cos=None, use_basic_aug=True):
        assert y is None or cos is None
        if training:
            aug = None
            ori = None
            cos_sim = None
            image_basic = None
            if y is not None:
                ori = x
                if use_basic_aug:
                    ori = torch.stack([self.pad_crop(x) for x in ori], dim=0)
                    ori = torch.stack([self.hflip(x) for x in ori], dim=0)
                image_basic = ori
                ori = self.normalize(ori)
                ori, features = self.net_forward(ori, training)

                if y.shape[-1] != ori.shape[-1]:
                    y = torch.zeros(ori.shape).to(y.device).scatter_(-1, y[..., None], 1.0)
                cos_sim = F.cosine_similarity(F.softmax(ori.detach(), dim=-1), y) ** (2 / np.log(y.shape[-1]))  # (N,)
                # cos_sim = 1. - torch.sqrt(torch.sum((F.softmax(ori.detach(), dim=-1) - y) ** 2, dim=-1) / 2.)  # Euclid / sqrt(2)

            else:
                aug = x
                if use_basic_aug:
                    aug = torch.stack([self.pad_crop(x) for x in aug], dim=0)
                    aug = torch.stack([self.hflip(x) for x in aug], dim=0)
                aug = self.sra(aug, cos)
                aug = self.cutout(aug)
                aug = self.normalize(aug)
                aug, features = self.net_forward(aug, training)
            return aug, ori, cos_sim, image_basic, features
        else:
            x = self.normalize(x)
            x = self.net_forward(x)
            return x, None


if __name__ == "__main__":
    from torchsummary import summary
    model = ShakeResNet(depth=26, w_base=96, label=10).cuda()
    summary(model, (3, 32, 32), device='cuda')
    x = torch.ones((8, 3, 32, 32)).cuda()
    print(model(x))