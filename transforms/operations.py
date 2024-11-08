import math
import numpy as np
import random
import collections

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import equalize as equalize_original
from torchvision.transforms import RandomCrop, RandomResizedCrop, RandomHorizontalFlip
"""
    The input of the following functions should be of range 0 to 255.
"""


def _trans_img_to_rank_4(images):
    """
    (H, W), (C, H, W), (N, C, H, W) to (N, C, H, W)
    """
    if len(images.shape) == 2:
        N = 1
        images = images[None, None]
    elif len(images.shape) == 3:
        N = 1
        images = images[None]
    else:
        N = images.shape[0]
    return N, images


def aug_transform(x, transforms):
    """
    :param x: (N, C, H, W)
    :param transforms: (N, 3, 3)
    :return: (N, H, W, 2). Transform based on the center of the image
    """
    transforms = transforms.to(x.dtype)
    x = F.grid_sample(
        x,
        transforms,
        padding_mode='zeros',
        mode='bilinear',
        align_corners=False,
    )
    return x


def blend(image1, image2, factor):
    if factor == 0.0:
        return image1
    if factor == 1.0:
        return image2

    ori_dtype = image1.dtype
    image1 = image1.to(torch.float64)
    image2 = image2.to(torch.float64)

    difference = image2 - image1
    scaled = factor * difference

    temp = image1 + scaled

    if factor > 0.0 and factor < 1.0:
        return temp.to(ori_dtype)
    return torch.clamp(temp, min=0.0, max=255.0).to(ori_dtype)


def grayscale(images):
    """
    images: (N, 3, H, W)
    """
    N, C, H, W = images.shape
    if C == 3:
        gray = (images[:, 0] * 0.299 + images[:, 1] * 0.587 + images[:, 2] * 0.114).reshape(N, 1, H, W)
    else:
        raise NotImplementedError('Channels for grayscale should be 3!')
    gray = torch.tile(gray, (1, 3, 1, 1))
    return gray


def shear_x(images, magnitude):
    """
    :param images: (N, C, H, W) or (C, H, W) or (H, W)
    """
    N, images = _trans_img_to_rank_4(images)
    theta = torch.tile(torch.tensor([[1., magnitude, magnitude],
                                    [0., 1., 0]]), (N, 1, 1))   # require translate "magnitude"
    theta = theta.to(images.device)
    grid = F.affine_grid(theta, images.shape, align_corners=False)
    return aug_transform(images, grid)


def shear_y(images, magnitude):
    N, images = _trans_img_to_rank_4(images)
    theta = torch.tile(torch.tensor([[1., 0., 0.],
                                    [magnitude, 1., magnitude]]), (N, 1, 1))   # require translate "magnitude"
    theta = theta.to(images.device)
    grid = F.affine_grid(theta, images.shape, align_corners=False)
    return aug_transform(images, grid)


def translate_x(images, magnitude):
    """
    Return the translated image (magnitude refers to the ratio of half of the full size)
    """
    N, images = _trans_img_to_rank_4(images)
    theta = torch.tile(torch.tensor([[1., 0., magnitude * 2],
                                    [0., 1., 0.]]), (N, 1, 1))
    theta = theta.to(images.device)
    grid = F.affine_grid(theta, images.shape, align_corners=False)
    return aug_transform(images, grid)


def translate_y(images, magnitude):
    N, images = _trans_img_to_rank_4(images)
    theta = torch.tile(torch.tensor([[1., 0., 0.],
                                    [0., 1., magnitude * 2]]), (N, 1, 1))
    theta = theta.to(images.device)
    grid = F.affine_grid(theta, images.shape, align_corners=False)
    return aug_transform(images, grid)


def _angle_to_rotation_matrix(angle):
    radians = angle * math.pi / 180.0
    radians = torch.Tensor([radians])
    cos_a = torch.cos(radians)
    sin_a = torch.sin(radians)
    return torch.stack([cos_a, -sin_a, sin_a, cos_a], dim=-1).view(1, 2, 2)


def _get_rotation_matrix(angle):
    # convert angle and apply scale
    if isinstance(angle, np.ndarray):
        angle = torch.from_numpy(angle)
    rotation_matrix = _angle_to_rotation_matrix(angle)
    M = torch.zeros((2, 3), dtype=torch.float64)
    M[:, 0:2] = rotation_matrix
    return M


def rotate(images, angle):
    """
    Positive value means anticlockwise rotation.
    """
    N, images = _trans_img_to_rank_4(images)
    trans_matrix = _get_rotation_matrix(angle)
    theta = torch.tile(trans_matrix, (N, 1, 1))
    theta = theta.to(images.device)
    grid = F.affine_grid(theta, images.shape, align_corners=False)
    return aug_transform(images, grid)


def posterize(images, bits):
    N, images = _trans_img_to_rank_4(images)
    ori_dtype = images.dtype
    def _left_shift(input, bits):
        # return (torch.bitwise_left_shift(input.to(torch.uint8), bits)).to(input.dtype)  # torch version >= 1.10
        return (input.to(torch.uint8) * (2 ** bits)).to(input.dtype)

    def _right_shift(input, bits):
        # return (torch.bitwise_right_shift(input.to(torch.uint8), bits)).to(input.dtype)  # torch version >= 1.10
        return (input.to(torch.uint8) / (2 ** bits)).to(input.dtype)

    images = _left_shift(_right_shift(images, bits), bits)

    return images.to(ori_dtype)


class straight_through_posterize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bits):
        x = posterize(x, bits)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def brightness(images, factor):
    N, images = _trans_img_to_rank_4(images)
    degenerate = torch.zeros_like(images)
    images = blend(degenerate, images, factor)
    return images


def color(images, factor):
    N, images = _trans_img_to_rank_4(images)
    if images.shape[1] == 3:
        # ratio for RGB: [0.299, 0.587, 0.114]
        degenerate = grayscale(images)
        images = blend(degenerate, images, factor)
    return images


def solarize(images, threshold):
    N, images = _trans_img_to_rank_4(images)
    images = torch.where(images < threshold, images, 255.0 - images)
    return images


def solarize_add(images, addition=0., threshold=128.):
    N, images = _trans_img_to_rank_4(images)
    ori_dtype = images.dtype
    images = images.to(torch.float64)
    images = torch.where(images < threshold, images + addition, images)
    return torch.clamp(images, 0., 255.).to(ori_dtype)


def sharpness(images, factor):
    N, images = _trans_img_to_rank_4(images)
    N, C, H, W = images.shape
    kernel = torch.tensor(
        [[1, 1, 1], [1, 5, 1], [1, 1, 1]], dtype=images.dtype).reshape(1, 1, 3, 3) / 13.
    kernel = torch.tile(kernel, (C, 1, 1, 1)).to(images.device)
    degenerate = torch.nn.functional.conv2d(
        images, kernel, stride=1, padding=1, groups=C)
    degenerate = torch.clamp(degenerate, 0.0, 255.0).to(images.dtype)
    mask = torch.ones_like(images).to(images.dtype)
    mask[:, :, 0, :] = 0
    mask[:, :, H - 1, :] = 0
    mask[:, :, :, 0] = 0
    mask[:, :, :, W - 1] = 0
    degenerate = torch.where(mask == 1, degenerate, images)
    images = blend(degenerate, images, factor)
    return images


def contrast(images, factor):
    N, images = _trans_img_to_rank_4(images)
    C = images.shape[1]
    if C == 3:
        degenerate = grayscale(images)
    else:
        # C = 1
        degenerate = images
    mean = torch.mean(degenerate, dim=(1, 2, 3), keepdim=True)
    degenerate = torch.ones_like(images) * mean
    images = blend(degenerate, images, factor)
    return images


def cutout(images, length):
    # Each batch has the same cutout
    N, images = _trans_img_to_rank_4(images)
    N, C, H, W = images.shape

    mask = np.ones((H, W), np.float32)

    y = np.random.randint(H)
    x = np.random.randint(W)
    half_len = int(length / 2)
    half_len_2 = length - half_len

    y1 = np.clip(y - half_len, 0, H)
    y2 = np.clip(y + half_len_2, 0, H)
    x1 = np.clip(x - half_len, 0, W)
    x2 = np.clip(x + half_len_2, 0, W)

    mask[y1:y2, x1:x2] = 0.

    mask = torch.from_numpy(mask).to(images.dtype).to(images.device)
    mask = mask.expand_as(images)
    images = images * mask

    return images


def autocontrast(images):
    N, images = _trans_img_to_rank_4(images)
    def _img_autocontrast(im):
        # img: (C, H, W)
        temp_c = []
        for c in range(im.shape[0]):
            lo = torch.min(im[c])
            hi = torch.max(im[c])
            if hi > lo:
                temp_c.append(255. * (im[c] - lo) / (hi - lo))
            else:
                temp_c.append(im[c])
        return torch.stack(temp_c, dim=0)

    images = torch.stack([_img_autocontrast(im) for im in images], dim=0)
    return images


def equalize(images):
    N, images = _trans_img_to_rank_4(images)
    ori_dtype = images.dtype

    images = images.to(torch.uint8)
    images = equalize_original(images)

    return images.to(ori_dtype)


class straight_through_equalize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = equalize(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def invert(images):
    N, images = _trans_img_to_rank_4(images)
    return 255.0 - images


class ShearX(torch.nn.Module):

    def __init__(self, min_val, max_val):
        super(ShearX, self).__init__()
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.op_name = 'ShearX'

    def _magnitude_to_arg(self, magnitude):
        if np.random.uniform() >= 0.5:
            magnitude = -magnitude
        return magnitude * self.max_val

    def forward(self, x, magnitude):
        x = shear_x(x, self._magnitude_to_arg(magnitude))
        return x


class ShearY(torch.nn.Module):

    def __init__(self, min_val, max_val):
        super(ShearY, self).__init__()
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.op_name = 'ShearY'

    def _magnitude_to_arg(self, magnitude):
        if np.random.uniform() >= 0.5:
            magnitude = -magnitude
        return magnitude * self.max_val

    def forward(self, x, magnitude):
        x = shear_y(x, self._magnitude_to_arg(magnitude))
        return x


class TranslateX(torch.nn.Module):

    def __init__(self, min_val, max_val):
        super(TranslateX, self).__init__()
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.op_name = 'TranslateX'

    def _magnitude_to_arg(self, magnitude):
        if np.random.uniform() >= 0.5:
            magnitude = -magnitude
        return magnitude * self.max_val

    def forward(self, x, magnitude):
        x = translate_x(x, self._magnitude_to_arg(magnitude))
        return x


class TranslateY(torch.nn.Module):

    def __init__(self, min_val, max_val):
        super(TranslateY, self).__init__()
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.op_name = 'TranslateY'

    def _magnitude_to_arg(self, magnitude):
        if np.random.uniform() >= 0.5:
            magnitude = -magnitude
        return magnitude * self.max_val

    def forward(self, x, magnitude):
        x = translate_y(x, self._magnitude_to_arg(magnitude))
        return x


class Rotate(torch.nn.Module):

    def __init__(self, min_val, max_val):
        super(Rotate, self).__init__()
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.op_name = 'Rotate'

    def _magnitude_to_arg(self, magnitude):
        if np.random.uniform() >= 0.5:
            magnitude = -magnitude
        return magnitude * self.max_val

    def forward(self, x, magnitude):
        x = rotate(x, self._magnitude_to_arg(magnitude))
        return x


class Brightness(torch.nn.Module):

    def __init__(self, min_val, max_val):
        super(Brightness, self).__init__()
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.op_name = 'Brightness'

    def _magnitude_to_arg(self, magnitude):
        if np.random.uniform() >= 0.5:
            magnitude = -magnitude
        return (self.max_val - 1.0) * magnitude + 1.0

    def forward(self, x, magnitude):
        x = brightness(x, self._magnitude_to_arg(magnitude))
        return x


class Color(torch.nn.Module):

    def __init__(self, min_val, max_val):
        super().__init__()
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.op_name = 'Color'

    def _magnitude_to_arg(self, magnitude):
        if np.random.uniform() >= 0.5:
            magnitude = -magnitude
        return (self.max_val - 1.0) * magnitude + 1.0

    def forward(self, x, magnitude):
        x = color(x, self._magnitude_to_arg(magnitude))
        return x


class Sharpness(torch.nn.Module):

    def __init__(self, min_val, max_val):
        super(Sharpness, self).__init__()
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.op_name = 'Sharpness'

    def _magnitude_to_arg(self, magnitude):
        if np.random.uniform() >= 0.5:
            magnitude = -magnitude
        return (self.max_val - 1.0) * magnitude + 1.0

    def forward(self, x, magnitude):
        x = sharpness(x, self._magnitude_to_arg(magnitude))
        return x


class Contrast(torch.nn.Module):

    def __init__(self, min_val, max_val):
        super(Contrast, self).__init__()
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.op_name = 'Contrast'

    def _magnitude_to_arg(self, magnitude):
        if np.random.uniform() >= 0.5:
            magnitude = -magnitude
        return (self.max_val - 1.0) * magnitude + 1.0

    def forward(self, x, magnitude):
        x = contrast(x, self._magnitude_to_arg(magnitude))
        return x


class Cutout(torch.nn.Module):

    def __init__(self, min_val, max_val):
        super(Cutout, self).__init__()
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.op_name = 'Cutout'

    def _magnitude_to_arg(self, h, magnitude):
        magnitude = (self.max_val - self.min_val) * magnitude + self.min_val
        length = torch.Tensor([magnitude * h]).to(torch.int32)
        return length

    def forward(self, x, magnitude):
        x = cutout(x, self._magnitude_to_arg(x.shape[2], magnitude))
        return x


class Solarize(torch.nn.Module):

    def __init__(self, min_val, max_val):
        super(Solarize, self).__init__()
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.op_name = 'Solarize'

    def _magnitude_to_arg(self, magnitude):
        threshold = (self.max_val - self.min_val) * magnitude + self.min_val
        return threshold

    def forward(self, x, magnitude):
        x = solarize(x, self._magnitude_to_arg(magnitude))
        return x


class SolarizeAdd(torch.nn.Module):

    def __init__(self, min_val, max_val):
        super(SolarizeAdd, self).__init__()
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.op_name = 'SolarizeAdd'

    def _magnitude_to_arg(self, magnitude):
        addition = (self.max_val - self.min_val) * magnitude + self.min_val
        return addition

    def forward(self, x, magnitude):
        x = solarize_add(x, self._magnitude_to_arg(magnitude))
        return x


class Posterize(torch.nn.Module):

    def __init__(self, min_val, max_val):
        super(Posterize, self).__init__()
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.op_name = 'Posterize'

    def _magnitude_to_arg(self, magnitude):
        # return torch.tensor((self.max_val - self.min_val) * magnitude + self.min_val, requires_grad=True).to(torch.int32)   # 类似原来的实现
        return (self.max_val - self.min_val) * magnitude + self.min_val                     # 当前计算方法特有

    def forward(self, x, magnitude):
        bits = self._magnitude_to_arg(magnitude)   # 类似原来的实现
        x = straight_through_posterize.apply(x, bits)
        # x = posterize(x, bits)
        return x


class Equalize(torch.nn.Module):

    def __init__(self, min_val, max_val):
        super(Equalize, self).__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.op_name = 'Equalize'

    def forward(self, x, magnitude):
        x = straight_through_equalize.apply(x)
        # x = equalize(x)
        return x


class AutoContrast(torch.nn.Module):
    def __init__(self, min_val, max_val):
        super(AutoContrast, self).__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.op_name = 'AutoContrast'

    def forward(self, x, magnitude):
        x = autocontrast(x)
        return x


class Invert(torch.nn.Module):

    def __init__(self, min_val, max_val):
        super(Invert, self).__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.op_name = 'Invert'

    def forward(self, x, magnitude):
        x = invert(x)
        return x


class Identity(torch.nn.Module):
    def __init__(self, min_val, max_val):
        super(Identity, self).__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.op_name = 'Identity'

    def forward(self, x, magnitude):
        return x


class HFlip(torch.nn.Module):
    def __init__(self, min_val=None, max_val=None, p=0.5):
        super(HFlip, self).__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.p = p
        self.hflip = RandomHorizontalFlip(p=p)
        self.op_name = 'HFlip'

    def forward(self, x, magnitude=None):
        x = self.hflip(x)
        return x


class PadCrop(torch.nn.Module):
    def __init__(self, min_val=None, max_val=None, size=32, padding=4):
        super(PadCrop, self).__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.padding = padding
        self.size = size
        self.pad_crop = RandomCrop(size, padding=padding)
        self.op_name = 'PadCrop'

    def forward(self, x, magnitude=None):
        x = self.pad_crop(x)
        return x


class ResizeCrop(torch.nn.Module):
    def __init__(self, min_val=None, max_val=None, size=224, scale=(0.08, 1.0), ratio=(0.75, 1.33)):
        super(ResizeCrop, self).__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.scale = scale
        self.resize_crop = RandomResizedCrop(size, scale=scale, ratio=ratio)
        self.ratio = ratio
        self.op_name = 'ResizeCrop'

    def forward(self, x, magnitude=None):
        x = self.resize_crop(x)
        return x
        

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x = torch.arange(64).reshape(1, 8, 8).to(torch.float64)
    x.requires_grad = True
    x = torch.tile(x, (3, 1, 1))
    x[0] = 0
    x[0, 6, 6] = 255
    # x = torch.randn(8, 8, requires_grad=True)


    # print(x.detach().numpy().transpose(1, 2, 0).shape)
    # plt.imshow(x.detach().numpy().transpose(1, 2, 0) / 255.)
    # plt.show()

    # out = shear_x(x, 0.9)
    # out = translate_y(x, 0.25)
    # out = rotate(x, 30)
    # out = posterize(x, 2)
    # print(out.shape)

    # func = Posterize(0, 8)
    # out = func(x, 0.4)
    # print(out)
    # out.backward(torch.ones_like(out))
    # print(x.grad)

    # func = Test()
    # out = func(x)
    # out.backward()
    # print(x.grad)

    # out = brightness(x, 0.3)
    # out = color(x, 0.3)
    # out = solarize(x, 32.1)
    # out = solarize_add(x, 32.1)
    # out = sharpness(x, 0.3)
    # out = contrast(x, 0.3)
    # out = cutout(x, 2)
    # out = autocontrast(x)
    # # out = x[None]
    #
    #
    # out = out.detach().numpy().transpose(0, 2, 3, 1)
    # print(out.shape)
    # print(out)
    #
    #
    # plt.imshow(out[0] / 255.)
    # plt.show()


    # print(111111111111111111111111)
    # # func = Posterize2(0, 8)
    # # out = func(x, 0.4)
    # # print(out)
    # # out.backward(torch.ones_like(out))
    # # print(x.grad)
    #
    # func = Solarize(0, 256)
    # func = ShearY(-0.3, 0.3)
    # func = TranslateX(-0.3, 0.3)
    func = Rotate(-30, 30)
    # func = Brightness(0.1, 1.9)
    # func = Color(0.1, 1.9)
    # func = Sharpness(0.1, 1.9)
    # func = Contrast(0.1, 1.9)
    # func = Cutout(0, 0.3)
    # func = Solarize(0, 255)
    # func = Posterize(0, 4)
    # func = Equalize(0., 1.)
    # func = AutoContrast(0., 1.)
    # func = SolarizeAdd(0, 110)
    out = func(x, 3)
    print(out)
    out.backward(torch.ones_like(out))
    print(x.grad)

    #
    # # out = brightness(x, 0.3)
    #
    #
    out = out.detach().numpy().transpose(0, 2, 3, 1)
    # print(out.shape)
    # print(out / 255.)
    plt.imshow((out[0]) / 255.)
    plt.show()