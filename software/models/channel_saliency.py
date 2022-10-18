import torch.nn as nn
import math
import torch
from torch.autograd import Variable


def expand(x, group_size):
    bs, vec_size = x.shape
    return x.unsqueeze(dim=-1).expand(bs, vec_size, group_size).reshape(bs, vec_size*group_size)


def get_pooling(method, full):
    if method == "max":
        return nn.AdaptiveMaxPool2d(1) if full else MaskedMaxPooling()
    else:
        return nn.AdaptiveAvgPool2d(1) if full else MaskedAvePooling()


class MaskedAvePooling(nn.Module):
    def __init__(self, size=1):
        super(MaskedAvePooling, self).__init__()
        self.pooling = nn.AdaptiveAvgPool2d(size)

    def forward(self, x, mask):
        mask = mask.hard
        if mask.shape[-1] != x.shape[-1] and mask.shape[-2] != x.shape[-2]:
            return self.pooling(x)
        pooled_feat = self.pooling(x * mask.expand_as(x))
        total_pixel_num = mask.shape[-1] * mask.shape[-2]
        active_pixel_num = mask.view(x.shape[0], -1).sum(dim=1)
        active_mask = active_pixel_num.unsqueeze(dim=1).unsqueeze(dim=1).unsqueeze(dim=1).expand_as(pooled_feat) + 1e-4
        return (pooled_feat * total_pixel_num)/active_mask


class MaskedMaxPooling(nn.Module):
    def __init__(self, size=1):
        super(MaskedMaxPooling, self).__init__()
        self.pooling = nn.AdaptiveAvgPool2d(size)

    def forward(self, feat, mask):
        if mask is None:
            return self.pooling(feat)
        else:
            return self.pooling(feat * mask.expand_as(feat))


class ChannelVectorUnit(nn.Module):
    def __init__(self, in_channels, out_channels, group_size=1, pooling_method="ave", channel_budget=1.0,
                 channel_stage=[-1], full_feature=False, device="cuda:0", **kwargs):
        super(ChannelVectorUnit, self).__init__()
        self.device = device
        self.full_feature = full_feature
        self.pooling = get_pooling(pooling_method, full_feature)
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.group_size = group_size
        # assert out_channels % group_size == 0, "The channels are not grouped with the same size"
        self.sigmoid = nn.Sigmoid()
        self.channel_saliency_predictor = nn.Linear(in_channels, out_channels//group_size)
        nn.init.kaiming_normal_(self.channel_saliency_predictor.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.channel_saliency_predictor.bias, 1.)
        self.sparsity = channel_budget
        self.target_stage = channel_stage

    def forward(self, x, meta):
        if meta["stage_id"] not in self.target_stage:
            self.using = False
            return torch.ones(x.shape[0], self.out_channels).to(device=self.device), meta
        self.using = True
        x = self.pooling(meta["saliency_mask"], meta["masks"][-1]["std"]) if not self.full_feature else self.pooling(x)
        x = x.view(x.shape[0], -1)
        x = self.channel_saliency_predictor(x)
        x = self.sigmoid(x)
        meta["lasso_sum"] += torch.mean(torch.sum(x, dim=-1))
        x = self.winner_take_all(x.clone())
        meta["channel_prediction"][(meta["stage_id"], meta["block_id"])] = x
        x = self.expand(x)
        return x, meta

    def expand(self, x):
        bs, vec_size = x.shape
        return x.unsqueeze(dim=-1).expand(bs, vec_size, self.group_size).reshape(bs, vec_size*self.group_size)

    def winner_take_all(self, x):
        if self.sparsity >= 1.0:
            return x
        elif x.size(-1) == 1:
            return (x > -1).int()
        else:
            k = math.ceil((1 - self.sparsity) * x.size(-1))
            inactive_idx = (-x).topk(k)[1]
            zero_filtered = x.scatter_(1, inactive_idx, 0)
            return (zero_filtered > 0).int()

    def get_flops(self):
        return 0 if not self.using else self.in_channels * self.out_channels / self.group_size


def conv_forward(conv_module, x, inp_vec=None, out_vec=None, forward=True):
    conv_module.__input_ratio__ = vector_ratio(inp_vec)
    conv_module.__output_ratio__ = vector_ratio(out_vec)
    if forward:
        return conv_module(x)


def bn_relu_foward(bn_module, relu_module, x, vector=None):
    bn_module.__output_ratio__ = vector_ratio(vector)
    relu_module.__output_ratio__ = vector_ratio(vector)
    if relu_module is not None:
        relu_module.vector = vector


def channel_process(x, vector):
    if len(vector.shape) != 2:
        return x * vector
    else:
        return x * vector.unsqueeze(-1).unsqueeze(-1).expand_as(x)


def vector_ratio(vector):
    if vector is None:
        return 1
    return torch.true_divide(vector.sum(), vector.numel()).tolist()

