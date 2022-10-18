import torch
import torch.nn as nn


class Mask:
    '''
    Class that holds the mask properties

    hard: the hard/binary mask (1 or 0), 4-dim tensor
    soft (optional): the float mask, same shape as hard
    active_positions: the amount of positions where hard == 1
    total_positions: the total amount of positions
                        (typically batch_size * output_width * output_height)
    '''

    def __init__(self, hard, soft=None):
        assert hard.dim() == 4
        assert hard.shape[1] == 1
        assert soft is None or soft.shape == hard.shape

        self.hard = hard
        self.active_positions = torch.sum(hard)  # this must be kept backpropagatable!
        self.total_positions = hard.numel()
        self.soft = soft

        self.flops_per_position = 0

    def size(self):
        return self.hard.shape

    def __repr__(self):
        return f'Mask with {self.active_positions}/{self.total_positions} positions, and {self.flops_per_position} accumulated FLOPS per position'


class MaskUnit(nn.Module):
    '''
    Generates the mask and applies the gumbel softmax trick
    '''

    def __init__(self, channels, out_height, out_width, stride=1, dilate_stride=1, mask_kernel=1, budget=0.5, **kwargs):
        super(MaskUnit, self).__init__()
        self.flops = mask_kernel * mask_kernel * out_height * out_width * channels
        self.maskconv = Squeeze(channels=channels, stride=stride, mask_kernel=mask_kernel)
        self.gumbel = Gumbel(**kwargs)
        self.stride = stride
        self.maxpool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x, meta):
        bs, _, w, h = x.shape
        soft = self.maskconv(x)
        hard = self.gumbel(soft, meta['gumbel_temp'], meta['gumbel_noise'])
        mask = Mask(hard, soft)

        if meta["stride"] == 2:
            m = {'std': Mask(self.maxpool(hard)), 'dilate': mask}
        else:
            m = {'std': mask, 'dilate': mask}
        meta['masks'].append(m)
        return m

    def get_flops(self):
        return torch.tensor(self.flops)


class Gumbel(nn.Module):
    '''
    Returns differentiable discrete outputs. Applies a Gumbel-Softmax trick on every element of x.
    '''

    def __init__(self, eps=1e-8, **kwargs):
        super(Gumbel, self).__init__()
        self.eps = eps
    
    def forward(self, x, gumbel_temp=1.0, gumbel_noise=True):
        if not self.training:  # no Gumbel noise during inference
            return (x >= 0).float()

        if gumbel_noise:
            eps = self.eps
            U1, U2 = torch.rand_like(x), torch.rand_like(x)
            g1, g2 = -torch.log(-torch.log(U1 + eps) + eps), - \
                torch.log(-torch.log(U2 + eps) + eps)
            x = x + g1 - g2

        soft = torch.sigmoid(x / gumbel_temp)
        hard = ((soft >= 0.5).float() - soft).detach() + soft
        assert not torch.any(torch.isnan(hard))
        return hard


class Squeeze(nn.Module):
    def __init__(self, channels, stride=1, mask_kernel=3, bias=2):
        padding = 1 if mask_kernel == 3 else 0
        super(Squeeze, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, 1, stride=stride, kernel_size=mask_kernel, padding=padding, bias=True)
        nn.init.constant_(self.conv.bias, bias)

    def forward(self, x):
        b, c, _, _ = x.size()
        return self.conv(x)


