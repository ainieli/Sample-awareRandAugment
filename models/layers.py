import torch
from torch.nn import Conv2d as Conv, Linear as Lin, BatchNorm1d, BatchNorm2d, ReLU, Sequential, Module, \
    AvgPool2d, MaxPool2d
import torch.nn.functional as F


def _init_weights(module):
    if isinstance(module, Lin):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)
        torch.nn.init.zeros_(module.bias)
    elif isinstance(module, Conv):
        torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, BatchNorm2d):
        torch.nn.init.ones_(module.weight)
        torch.nn.init.zeros_(module.bias)


def make_layer(block, in_channels, channels, blocks, stride=1, **kwargs):
    layers = [block(in_channels, channels, stride=stride, **kwargs)]
    in_channels = channels * block.expansion
    for i in range(1, blocks):
        layers.append(block(in_channels, channels, stride=1, **kwargs))
    return Sequential(*layers)


def calc_same_padding(kernel_size, dilation=(1, 1)):
    kh, kw = kernel_size
    dh, dw = dilation
    ph = (kh + (kh - 1) * (dh - 1) - 1) // 2
    pw = (kw + (kw - 1) * (dw - 1) - 1) // 2
    padding = (ph, pw)
    return padding


def NormAct(channels, norm='bn', act='relu'):
    layers = []
    if norm:
        norm_l = Norm(channels, norm)
        _init_weights(norm_l)
        layers.append(norm_l)
    if act:
        layers.append(Act(act))
    if len(layers) == 1:
        return layers[0]
    else:
        return Sequential(*layers)


def Norm(channels=None, type='bn', affine=True, track_running_stats=True):
    if type == 'bn':
        norm_l = BatchNorm2d(channels, affine=affine, track_running_stats=track_running_stats)
        _init_weights(norm_l)
    else:
        raise ValueError("Unsupported normalization type: %s" % type)
    return norm_l


def Act(type='relu'):
    if type == 'relu':
        return ReLU()
    else:
        raise ValueError("Unsupported activation type: %s" % type)


def Conv2d(in_channels,
           out_channels,
           kernel_size,
           stride=1,
           padding='same',
           groups=1,
           dilation=1,
           bias=None,
           norm=None,
           act=None):

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    if isinstance(padding, str):
        assert padding == 'same'

    if padding == 'same':
        padding = calc_same_padding(kernel_size, dilation)
    if bias is None:
        bias = norm is None

    layers = []
    conv = Conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
             padding=padding, groups=groups, dilation=dilation, bias=bias)
    _init_weights(conv)
    layers.append(conv)
    if norm is not None:
        norm_l = Norm(out_channels, norm)
        _init_weights(norm_l)
        layers.append(norm_l)
    if act is not None:
        layers.append(Act(act))

    if len(layers) == 1:
        return layers[0]
    else:
        return Sequential(*layers)


def Linear(in_channels, out_channels, bias=True, norm=None, act=None):
    layers = []
    if bias is None:
        bias = norm is None
    fc = Lin(in_channels, out_channels, bias=bias)
    _init_weights(fc)

    if norm == 'bn':
        norm_l = BatchNorm1d(out_channels)
        _init_weights(norm_l)
        layers.append(norm_l)
    if act is not None:
        layers.append(Act(act))
    layers = [fc] + layers
    if len(layers) == 1:
        return layers[0]
    else:
        return Sequential(*layers)


def Pool2d(kernel_size, stride, padding='same', type='avg', ceil_mode=True):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if padding == 'same':
        padding = calc_same_padding(kernel_size)
    if type == 'avg':
        return AvgPool2d(kernel_size, stride, padding, ceil_mode=ceil_mode, count_include_pad=False)
    elif type == 'max':
        return MaxPool2d(kernel_size, stride, padding, ceil_mode=ceil_mode)
    else:
        raise NotImplementedError("No activation named %s" % type)


class GlobalAvgPool(Module):

    def __init__(self, keep_dim=False):
        super().__init__()
        self.keep_dim = keep_dim

    def forward(self, x):
        x = F.adaptive_avg_pool2d(x, 1)
        if not self.keep_dim:
            x = x.view(x.size(0), -1)
        return x


