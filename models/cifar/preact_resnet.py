import numpy as np

import torch
import torch.nn.functional as F

from models.layers import make_layer, Conv2d, NormAct, Linear, GlobalAvgPool
from models.preact_resblocks import BasicBlock

from transforms.SRA import SampleAwareRandAugment as SRA
from transforms import Cutout, Normalize
from transforms.operations import HFlip, PadCrop


class ResNet(torch.nn.Module):

    def __init__(self, depth, k, block, num_classes=10, channels=(16, 16, 32, 64), dropout=0,
                 aug_depth=2,
                 resolution=32, augment_space='RA', p_min_t=0.2, p_max_t=0.8,
                 cutout_len=16,
                 norm_mean=[0.491, 0.482, 0.447], norm_std=[0.247, 0.243, 0.262]
                 ):
        super(ResNet, self).__init__()
        layers = [(depth - 4) // 6] * 3
        channels = (channels[0],) + tuple(c * k for c in channels[1:])

        stem_channels, *channels = channels

        self.stem = Conv2d(3, stem_channels, kernel_size=3)
        c_in = stem_channels

        strides = [1, 2, 2]
        for i, (c, n, s) in enumerate(zip(channels, layers, strides)):
            layer = make_layer(block, c_in, c, n, s,
                               dropout=dropout)
            c_in = c * block.expansion
            setattr(self, "layer" + str(i + 1), layer)

        self.norm_act = NormAct(c_in)
        self.avgpool = GlobalAvgPool()
        self.fc = Linear(c_in, num_classes)

        self.aug_depth = aug_depth
        self.sra = SRA(depth=aug_depth,
                           resolution=resolution, augment_space=augment_space,
                           p_min_t=p_min_t, p_max_t=p_max_t
                           )
        self.pad_crop = PadCrop(size=resolution, padding=4)
        self.hflip = HFlip()
        self.cutout = Cutout(cutout_len)
        self.normalize = Normalize(norm_mean, norm_std)

    def net_forward(self, x, training=False):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.norm_act(x)
        x = self.avgpool(x)
        if training == True:
            features = x
            x = self.fc(x)
            return x, features
        else:
            x = self.fc(x)
            return x

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
                cos_sim = F.cosine_similarity(F.softmax(ori.detach(), dim=-1), y) ** (2 / np.log(y.shape[-1]))   # (N,)
                #cos_sim = 1. - torch.sqrt(torch.sum((F.softmax(ori.detach(), dim=-1) - y) ** 2, dim=-1) / 2.)  # Euclid / sqrt(2)

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


def WRN_40_2(**kwargs):
    return ResNet(depth=40, k=2, block=BasicBlock, **kwargs)


def WRN_28_10(**kwargs):
    return ResNet(depth=28, k=10, block=BasicBlock, **kwargs)


if __name__ == "__main__":
    from torchsummary import summary
    from loss import DRASearchLoss

    model = WRN_40_2().cuda()
    cri = DRASearchLoss()
    # summary(model, (3, 32, 32), device='cpu')
    # summary(model, (3, 32, 32), device='cuda')

    x = torch.ones((8, 3, 32, 32)).cuda()
    y = torch.arange(8).cuda()
    out = model(x, training=True, y=y)

    loss = cri(out, y, search=True)
