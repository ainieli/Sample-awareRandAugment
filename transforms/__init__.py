import torch

from transforms.operations import cutout, _trans_img_to_rank_4


def normalize(x, mean, std):
    N, C, H, W = x.shape
    mean = torch.tensor(mean).to(x.dtype).to(x.device).reshape(1, C, 1, 1)
    std = torch.tensor(std).to(x.dtype).to(x.device).reshape(1, C, 1, 1)
    x = (x - mean) / std
    return x


class Cutout(torch.nn.Module):
    def __init__(self, length):
        super(Cutout, self).__init__()
        self.length = length

    def forward(self, x):
        ori_shape = x.shape
        N, x = _trans_img_to_rank_4(x)
        if self.length == 0:
            return x
        x_list = [cutout(x[n], self.length) for n in range(N)]
        x = torch.cat(x_list, dim=0).reshape(ori_shape)
        return x


class Normalize(torch.nn.Module):
    # default mean & std for CIFAR
    def __init__(self, mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        x = normalize(x, self.mean, self.std)
        return x

# from https://github.com/VDIGPKU/DADA/blob/master/search_relax/operation.py#L200
class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return

        N = img.shape[0]
        alpha = img.new().resize_((N, 3)).normal_(0, self.alphastd).to(img.device)
        imgs = []
        for i in range(N):
            rgb = self.eigvec.type_as(img).clone().to(img.device) \
                .mul(alpha[i].view(1, 3).expand(3, 3)) \
                .mul(self.eigval.view(1, 3).expand(3, 3).to(img.device)) \
                .sum(1).squeeze()
            s_img = img[i].add(rgb.view(3, 1, 1).expand_as(img[i]))
            imgs.append(s_img)

        return torch.stack(imgs, dim=0)


if __name__ == "__main__":
    x = torch.ones((2, 3, 8, 8)).cuda()
    # x[1] = x[1] * 2
    # cut = Cutout(4)
    # print(cut(x).shape)

    # mean = [0.491, 0.482, 0.447]
    # std = [0.247, 0.243, 0.262]
    # norm = Normalize(mean, std)
    # print(norm(x))

    from dataset import _IMAGENET_PCA
    l = Lighting(0.1, _IMAGENET_PCA['eigval'], _IMAGENET_PCA['eigvec'])
    print(l(x))
