import collections
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from transforms.operations import ShearX, ShearY, TranslateX, TranslateY, Rotate, Brightness, Color, \
    Sharpness, Contrast, Solarize, Posterize, Equalize, AutoContrast, Identity


CANDIDATE_OPS_DICT_32 = CANDIDATE_OPS_DICT_224 = collections.OrderedDict({
            'ShearX': ShearX(-0.3, 0.3),
            'ShearY': ShearY(-0.3, 0.3),
            'TranslateX': TranslateX(-0.45, 0.45),
            'TranslateY': TranslateY(-0.45, 0.45),
            'Rotate': Rotate(-30, 30),
            'Brightness': Brightness(0.1, 1.9),
            'Color': Color(0.1, 1.9),
            'Sharpness': Sharpness(0.1, 1.9),
            'Contrast': Contrast(0.1, 1.9),
            'Solarize': Solarize(0, 256),
            'Posterize': Posterize(0, 4),
            'Equalize': Equalize(None, None),
            'AutoContrast': AutoContrast(None, None),
            'Identity': Identity(None, None),
        })

RA_OP_NAME = CANDIDATE_OPS_DICT_32.keys()

RA_shearx = [
    'ShearY', 'TranslateX', 'TranslateY', 'Rotate',
    'Brightness', 'Color', 'Sharpness', 'Contrast',
    'Solarize', 'Posterize', 'Equalize', 'AutoContrast',
    'Identity'
]

RA_sheary = [
    'ShearX', 'TranslateX', 'TranslateY', 'Rotate',
    'Brightness', 'Color', 'Sharpness', 'Contrast',
    'Solarize', 'Posterize', 'Equalize', 'AutoContrast',
    'Identity'
]


RA_translatex = [
    'ShearX', 'ShearY', 'TranslateY', 'Rotate',
    'Brightness', 'Color', 'Sharpness', 'Contrast',
    'Solarize', 'Posterize', 'Equalize', 'AutoContrast',
    'Identity'
]

RA_translatey = [
    'ShearX', 'ShearY', 'TranslateX', 'Rotate',
    'Brightness', 'Color', 'Sharpness', 'Contrast',
    'Solarize', 'Posterize', 'Equalize', 'AutoContrast',
    'Identity'
]

RA_rotate = [
    'ShearX', 'ShearY', 'TranslateX', 'TranslateY',
    'Brightness', 'Color', 'Sharpness', 'Contrast',
    'Solarize', 'Posterize', 'Equalize', 'AutoContrast',
    'Identity'
]

RA_brightness = [
    'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate',
    'Color', 'Sharpness', 'Contrast',
    'Solarize', 'Posterize', 'Equalize', 'AutoContrast',
    'Identity'
]

RA_color = [
    'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate',
    'Brightness', 'Sharpness', 'Contrast',
    'Solarize', 'Posterize', 'Equalize', 'AutoContrast',
    'Identity'
]

RA_sharpness = [
    'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate',
    'Brightness', 'Color', 'Contrast',
    'Solarize', 'Posterize', 'Equalize', 'AutoContrast',
    'Identity'
]

RA_contrast = [
    'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate',
    'Brightness', 'Color', 'Sharpness',
    'Solarize', 'Posterize', 'Equalize', 'AutoContrast',
    'Identity'
]

RA_solarize = [
    'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate',
    'Brightness', 'Color', 'Sharpness', 'Contrast',
    'Posterize', 'Equalize', 'AutoContrast',
    'Identity'
]

RA_posterize = [
    'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate',
    'Brightness', 'Color', 'Sharpness', 'Contrast',
    'Solarize', 'Equalize', 'AutoContrast',
    'Identity'
]

RA_equalize = [
    'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate',
    'Brightness', 'Color', 'Sharpness', 'Contrast',
    'Solarize', 'Posterize', 'AutoContrast',
    'Identity'
]

RA_autocontrast = [
    'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate',
    'Brightness', 'Color', 'Sharpness', 'Contrast',
    'Solarize', 'Posterize', 'Equalize',
    'Identity'
]

RA_identity = [
    'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate',
    'Brightness', 'Color', 'Sharpness', 'Contrast',
    'Solarize', 'Posterize', 'Equalize', 'AutoContrast'
]


RA_SPACE = {
    'RA': RA_OP_NAME,
    'RA-shearx': RA_shearx,
    'RA-sheary': RA_sheary,
    'RA-translatex': RA_translatex,
    'RA-translatey': RA_translatey,
    'RA-rotate': RA_rotate,
    'RA-brightness': RA_brightness,
    'RA-color': RA_color,
    'RA-sharpness': RA_sharpness,
    'RA-contrast': RA_contrast,
    'RA-solarize': RA_solarize,
    'RA-posterize': RA_posterize,
    'RA-equalize': RA_equalize,
    'RA-autocontrast': RA_autocontrast,
    'RA-identity': RA_identity,
}


def one_hot(value, n_elements, axis=-1):
    one_h = torch.zeros(n_elements).scatter_(axis, value[..., None], 1.0).to(value.device)
    return one_h


def gumbel_softmax(logits, tau=1.0, hard=False, axis=-1):
    u = np.random.uniform(low=0., high=1., size=logits.shape)
    gumbel = torch.from_numpy(-np.log(-np.log(u))).to(logits.dtype).to(logits.device)
    gumbel = (logits + gumbel) / tau
    y_soft = F.softmax(gumbel, dim=-1)
    if hard:
        index = torch.argmax(y_soft, dim=axis).to(y_soft.device)
        y_hard = one_hot(index, y_soft.shape[axis], axis)
        ret = y_hard + y_soft - y_soft.detach()
    else:
        ret = y_soft
    return ret


class SampleAwareRandAugment(torch.nn.Module):

    def __init__(self, depth=2,
                 resolution=224, augment_space='RA', p_min_t=0.2, p_max_t=0.8):
        super(SampleAwareRandAugment, self).__init__()
        assert augment_space in RA_SPACE.keys()

        if resolution == 224:
            CANDIDATE_OPS_LIST = [CANDIDATE_OPS_DICT_224[op] for op in RA_SPACE[augment_space]]
        elif resolution == 32:
            CANDIDATE_OPS_LIST = [CANDIDATE_OPS_DICT_32[op] for op in RA_SPACE[augment_space]]
        else:
            raise NotImplementedError('The search space with resolution apart from 32 and 224 should be redefined!')

        self.candidate_ops = CANDIDATE_OPS_LIST
        num_candidate_ops = len(self.candidate_ops)
        self.sp_magnitudes_mean = None

        self.depth = depth
        self.num_candidate_ops = num_candidate_ops
        self.p_min_t = p_min_t
        self.p_max_t = p_max_t

    def learnable_params(self):
        return []

    def _apply_op(self, x, prob, s_op, mag_mean):
        magnitude = mag_mean

        # # for RA + std, std=0.5
        # eta = torch.randn(size=(), dtype=mag_mean.dtype)
        # magnitude = mag_mean + eta * 0.5
        # magnitude = torch.clamp(magnitude, 0., 1.)

        p = np.random.uniform(low=0., high=1.)
        if p < prob:
            x = self.candidate_ops[s_op](x, magnitude)
        return x

    def forward(self, x, cos_sim):
        # self.clip_value() # Shouldn't be used in the forward
                            # to avoid learnable tensors being non-leaf
        assert len(x.shape) == 4    # x: (N, C, H, W), float32, 0-1
        assert cos_sim.shape[-1] == x.shape[0]
        if torch.max(x) <= 1:
            x = x * 255
        x = x.to(torch.float32)

        if len(cos_sim.shape) == 1:
            cos_sim_list = [cos_sim] * self.depth           # "depth" layers share the same magnitude for each sample 
            cos_sim = torch.stack(cos_sim_list, dim=0)     # (depth, N)
        self.sp_magnitudes_mean = cos_sim

        x_batch = []
        for n in range(x.shape[0]):
            x_temp = x[n][None]
            for i in range(self.depth):
                j = torch.randint(low=0, high=self.num_candidate_ops, size=())
                p = np.random.uniform(low=self.p_min_t, high=self.p_max_t)
                mag = self.sp_magnitudes_mean[i, n]
                x_temp = self._apply_op(x_temp, p, j, mag)

            x_batch.append(x_temp)
        x = torch.cat(x_batch, dim=0)

        x = torch.clamp(x / 255., 0., 1.)
        return x


class SampleAwareRandAugment_DIFF(torch.nn.Module):

    def __init__(self, depth=2,
                 resolution=224, augment_space='RA', p_min_t=0.2, p_max_t=0.8, tau=1.0, learnable_w=True):
        super(SampleAwareRandAugment_DIFF, self).__init__()
        assert augment_space in RA_SPACE.keys()

        if resolution == 224:
            CANDIDATE_OPS_LIST = [CANDIDATE_OPS_DICT_224[op] for op in RA_SPACE[augment_space]]
        elif resolution == 32:
            CANDIDATE_OPS_LIST = [CANDIDATE_OPS_DICT_32[op] for op in RA_SPACE[augment_space]]
        else:
            raise NotImplementedError('The search space with resolution apart from 32 and 224 should be redefined!')

        self.candidate_ops = CANDIDATE_OPS_LIST
        num_candidate_ops = len(self.candidate_ops)
        # self.sp_weights = Variable(1e-3 * torch.ones(depth, num_candidate_ops), requires_grad=True)
        self.register_parameter('sp_weights', torch.nn.Parameter(1e-3 * torch.ones(depth, num_candidate_ops), requires_grad=True))
        self.sp_magnitudes_mean = None

        self.tau = tau
        self.depth = depth
        self.num_candidate_ops = num_candidate_ops
        self.p_min_t = p_min_t
        self.p_max_t = p_max_t
        self.learnable_w = learnable_w

    def learnable_params(self):
        return [
            self.sp_weights
        ]

    def _apply_op(self, x, prob, s_op, mag_mean):
        magnitude = mag_mean
        p = np.random.uniform(low=0., high=1.)
        if p < prob:
            x = self.candidate_ops[s_op](x, magnitude)
        return x

    def forward(self, x, cos_sim):
        # self.clip_value() # Shouldn't be used in the forward
                            # to avoid learnable tensors being non-leaf
        assert len(x.shape) == 4    # x: (N, C, H, W), float32, 0-1
        assert cos_sim.shape[-1] == x.shape[0]
        if torch.max(x) <= 1:
            x = x * 255
        x = x.to(torch.float32)

        if len(cos_sim.shape) == 1:
            cos_sim_list = [cos_sim] * self.depth           # "depth" layers share the same magnitude for each sample
            cos_sim = torch.stack(cos_sim_list, dim=0)     # (depth, N)
        self.sp_magnitudes_mean = cos_sim

        x_batch = []
        for n in range(x.shape[0]):
            x_temp = x[n][None]
            for i in range(self.depth):
                if self.learnable_w:
                    hardwts = gumbel_softmax(self.sp_weights[i], tau=self.tau, hard=True)
                else:
                    sampled_op_idx = torch.randint(low=0, high=self.num_candidate_ops, size=())
                    hardwts = one_hot(sampled_op_idx, self.num_candidate_ops)
                j = torch.argmax(hardwts)
                p = np.random.uniform(low=self.p_min_t, high=self.p_max_t)
                mag = self.sp_magnitudes_mean[i, n]
                x_temp = self._apply_op(x_temp, p, j, mag) * hardwts[j] + torch.sum(hardwts) - hardwts[j]

            x_batch.append(x_temp)
        x = torch.cat(x_batch, dim=0)

        x = torch.clamp(x / 255., 0., 1.)
        return x


class SampleAwareRandAugment_MAB(torch.nn.Module):

    def __init__(self, depth=2,
                 resolution=224, augment_space='RA', p_min_t=0.2, p_max_t=0.8, tau=1.0, learnable_w=True,
                 reward_decay=0.98, learning_rate=1e-2):
        super(SampleAwareRandAugment_MAB, self).__init__()
        assert augment_space in RA_SPACE.keys()

        if resolution == 224:
            CANDIDATE_OPS_LIST = [CANDIDATE_OPS_DICT_224[op] for op in RA_SPACE[augment_space]]
        elif resolution == 32:
            CANDIDATE_OPS_LIST = [CANDIDATE_OPS_DICT_32[op] for op in RA_SPACE[augment_space]]
        else:
            raise NotImplementedError('The search space with resolution apart from 32 and 224 should be redefined!')

        self.candidate_ops = CANDIDATE_OPS_LIST
        num_candidate_ops = len(self.candidate_ops)
        self.sp_weights = Variable(0 * torch.ones(depth, num_candidate_ops), requires_grad=True)
        # self.register_parameter('sp_weights', torch.nn.Parameter(0. * torch.ones(depth, num_candidate_ops), requires_grad=True))
        self.sp_magnitudes_mean = None

        self.tau = tau
        self.reward_decay = reward_decay
        self.learning_rate = learning_rate
        self.depth = depth
        self.num_candidate_ops = num_candidate_ops
        self.p_min_t = p_min_t
        self.p_max_t = p_max_t
        self.learnable_w = learnable_w

        if learnable_w:
            self.reward_op = None
            self.selected_op = None

    def learnable_params(self):
        return [
            self.sp_weights
        ]

    def _apply_op(self, x, prob, s_op, mag_mean):
        magnitude = mag_mean
        p = np.random.uniform(low=0., high=1.)
        if p < prob:
            x = self.candidate_ops[s_op](x, magnitude)
        return x

    def _cal_bin_cnts_op(self, batch_loss):
        bin_cnts = torch.zeros(self.depth, self.num_candidate_ops).to(batch_loss.device)
        for i in range(self.depth):
            bin_cnts[i] = torch.tensor(
                [torch.sum(torch.where(self.selected_op[i] == x, 1, 0)) for x in range(self.num_candidate_ops)])
        return bin_cnts

    def _cal_op_avg_pred(self, batch_loss):
        batch_loss = batch_loss.detach()
        pred_sum = torch.zeros(self.depth, self.num_candidate_ops).to(batch_loss.device)
        batch_pred = torch.exp(-batch_loss)
        for i, pred in enumerate(batch_pred):
            pred_sum[range(self.depth), self.selected_op[:, i]] += pred
        bin_cnts = self._cal_bin_cnts_op(batch_loss)
        mag_avg_pred = pred_sum / bin_cnts
        return mag_avg_pred

    def _cal_reward_op(self, batch_loss):
        batch_loss = batch_loss.detach()
        self.reward_op = self._cal_op_avg_pred(batch_loss)
        self.reward_op = torch.stack(
            [torch.where(torch.isnan(self.reward_op[x]), torch.full_like(self.reward_op[x], 0), self.reward_op[x]) for x
             in range(self.depth)], dim=0)  # replace NaN value to 0

    def update_w(self, batch_loss):
        batch_loss = batch_loss.detach()
        # V(a) ← V(a) + alpha * (reward + gamma * Vmax(a) - V(a))
        self.sp_weights = self.sp_weights.to(batch_loss.device)
        self.reward_op = self._cal_reward_op(batch_loss)
        # self.reward_op = self.reward_op.to(batch_loss.device)
        Vmax = torch.max(self.sp_weights, dim=-1)[0][:, None].repeat(1, self.num_candidate_ops)
        self.sp_weights = self.sp_weights + self.learning_rate * (self.reward_op + self.reward_decay * Vmax - self.sp_weights)

    def forward(self, x, cos_sim):
        # self.clip_value() # Shouldn't be used in the forward
                            # to avoid learnable tensors being non-leaf
        assert len(x.shape) == 4    # x: (N, C, H, W), float32, 0-1
        assert cos_sim.shape[-1] == x.shape[0]
        if torch.max(x) <= 1:
            x = x * 255
        x = x.to(torch.float32)

        if len(cos_sim.shape) == 1:
            cos_sim_list = [cos_sim] * self.depth           # "depth" layers share the same magnitude for each sample
            cos_sim = torch.stack(cos_sim_list, dim=0)     # (depth, N)
        self.sp_magnitudes_mean = cos_sim

        N = x.shape[0]
        selected_op = torch.zeros(self.depth, N)

        x_batch = []
        for n in range(x.shape[0]):
            x_temp = x[n][None]
            for i in range(self.depth):
                if self.learnable_w:
                    hardwts = gumbel_softmax(self.sp_weights[i], tau=self.tau, hard=True)
                else:
                    sampled_op_idx = torch.randint(low=0, high=self.num_candidate_ops, size=())
                    hardwts = one_hot(sampled_op_idx, self.num_candidate_ops)
                j = torch.argmax(hardwts)
                selected_op[i, n] = j
                p = np.random.uniform(low=self.p_min_t, high=self.p_max_t)
                mag = self.sp_magnitudes_mean[i, n]
                x_temp = self._apply_op(x_temp, p, j, mag) * hardwts[j] + torch.sum(hardwts) - hardwts[j]

            x_batch.append(x_temp)
        x = torch.cat(x_batch, dim=0)

        x = torch.clamp(x / 255., 0., 1.)
        self.selected_op = selected_op.long()
        return x