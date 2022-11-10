import torch
from models.channel_saliency import conv_forward, bn_relu_foward, channel_process, ChannelVectorUnit
from torch import nn
from models.utils import conv2d_out_dim, apply_mask, NoneMask, get_input_active_matrix
from .layers import *
from .maskunit import *


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None, sparse=False):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class InvertedResidualBlock(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, inp_h, inp_w, norm_layer=None, resolution_mask=False,
                 mask_block=False, final_activation="linear", downsample=False, before_residual=False,
                 dropout_ratio=0, dropout_stages=[-1], channel_budget=-1,budget=-1, group_size=1, device="cuda:0", **kwargs):
        super(InvertedResidualBlock, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.sparse = budget > 0
        self.channel_budget = channel_budget
        self.spatial_budget = budget
        self.use_res_connect = (self.stride == 1 and inp == oup) if downsample is None else True
        self.downsample = downsample
        self.inp_height = inp_h
        self.inp_width = inp_w
        self.out_height = conv2d_out_dim(inp_h, kernel_size=3, stride=stride, padding=1)
        self.out_width = conv2d_out_dim(inp_w, kernel_size=3, stride=stride, padding=1)
        self.inplanes, self.outplanes = inp, oup

        self.resolution_mask = resolution_mask
        self.mask_block = mask_block
        self.before_residual = before_residual
        hidden_dim = int(round(inp * expand_ratio))

        if self.sparse:
            if channel_budget >= 0:
                self.saliency = ChannelVectorUnit(in_channels=inp, out_channels=hidden_dim,
                                                  group_size=group_size, channel_budget=channel_budget, **kwargs)

        if self.sparse and self.use_res_connect:
            if not (self.resolution_mask and self.mask_block):
                    self.masker = MaskUnit(channels=inp, stride=stride, budget=budget, dilate_stride=1,
                                                   out_height=self.out_height, out_width=self.out_width, **kwargs)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.activation = nn.ReLU(inplace=True)

        self.conv_pw_1 = nn.Conv2d(inp, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(hidden_dim)

        self.conv3x3_dw = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim,
                                    bias=False)
        self.bn_dw = norm_layer(hidden_dim)

        self.conv_pw_2 = nn.Conv2d(hidden_dim, oup, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = norm_layer(oup)
        self.final_activation = final_activation

        self.dropout_ratio = dropout_ratio
        self.dropout_stages = dropout_stages

        flops_conv1_full = torch.Tensor([self.inp_height * self.inp_width * hidden_dim * inp])
        flops_conv2_full = torch.Tensor([9 * self.out_height * self.out_width * hidden_dim])
        flops_conv3_full = torch.Tensor([self.out_height * self.out_width * hidden_dim * oup])
        self.flops_downsample = torch.Tensor([self.outp_height*self.outp_width*self.outplanes*self.expansion*self.inplanes]
                                            ) if self.downsample is not None else torch.Tensor([0])
        self.flops_full = (flops_conv1_full + flops_conv2_full + flops_conv3_full + self.flops_downsample).to(device=device)

    def forward_basic(self, x, meta):
        x = self.activation(self.bn1(self.conv_pw_1(x)))
        x = self.activation(self.bn_dw(self.conv3x3_dw(x)))
        x = self.bn2(self.conv_pw_2(x))
        meta["saliency_mask"] = x
        return x, meta

    def get_flops(self, mask_c, mask_s):
        if not self.sparse:
            return self.flops_full
        channel_sum = mask_c.sum(1)
        spatial_sum = mask_s.sum((1,2,3))

        flops_conv1 = spatial_sum * channel_sum * self.inplanes
        flops_conv2 = 9 * spatial_sum * channel_sum
        flops_conv3 = spatial_sum * self.outplanes * channel_sum
        spatial_flops, channel_flops = torch.tensor(0), torch.tensor(0)
        if self.saliency.using:
            channel_flops = self.saliency.get_flops()
        if hasattr(self, "masker"):
            spatial_flops = self.masker.get_flops()
        flops_mask = (spatial_flops + channel_flops).cuda()
        # total
        flops = (flops_conv1+flops_conv2+flops_conv3+flops_mask).to(flops_conv1.device)

        return flops.cuda()

    def obtain_mask(self, x, meta):
        if self.resolution_mask:
            if self.mask_block:
                m = self.masker(x, meta)
            else:
                if self.input_resolution:
                    m = {"dilate": meta["masks"][-1]["std"], "std": meta["masks"][-1]["std"]}
                else:
                    m = meta["masks"][-1]
        else:
            m = self.masker(x, meta)
        return m

    def forward_channel_pruning(self, x, meta):

        vector, meta = self.saliency(x, meta)

        conv_forward(self.conv_pw_1, None, None, vector, forward=False)
        conv_forward(self.conv3x3_dw, None, None, vector, forward=False)
        conv_forward(self.conv_pw_2, None, vector, None, forward=False)

        x = self.activation(self.bn1(self.conv_pw_1(x)))
        x = channel_process(x, vector)
        x = self.activation(self.bn_dw(self.conv3x3_dw(x)))
        x = channel_process(x, vector)
        x = self.bn2(self.conv_pw_2(x))
        if self.before_residual:
            meta["saliency_mask"] = x
        return x, meta, vector

    def forward_block(self, inp):
        x, meta = inp
        if (not self.sparse) or (not self.use_res_connect):
            mask_s = torch.ones(x.shape[0], 1, self.out_height, self.out_width).cuda()
            if self.channel_budget == -1:
                x, meta = self.forward_basic(x, meta)
                meta["flops"] = torch.cat((meta["flops"], self.flops_full.repeat(x.shape[0]).unsqueeze(dim=0)))
            else:
                x, meta, mask_c = self.forward_channel_pruning(x, meta)
                meta["flops"] = torch.cat((meta["flops"], self.get_flops(mask_c, mask_s).unsqueeze(dim=0)))
        else:
            meta["stride"] = self.stride
            if self.channel_budget > 0:
                vector, meta = self.saliency(x, meta)
                conv_forward(self.conv_pw_1, None, None, vector, forward=False)
                conv_forward(self.conv3x3_dw, None, None, vector, forward=False)
                conv_forward(self.conv_pw_2, None, vector, None, forward=False)
            m = self.obtain_mask(x, meta)
            mask_dilate, mask = m['dilate'], m['std']

            x = conv1x1_mask(self.conv_pw_1, x, mask_dilate)
            x = bn_relu_mask(self.bn1, self.activation, x, mask_dilate)
            x = apply_mask(x, mask_dilate)
            if self.channel_budget > 0:
                x = channel_process(x, vector)
            x = conv3x3_dw_mask(self.conv3x3_dw, x, mask_dilate, mask)
            x = bn_relu_mask(self.bn_dw, self.activation, x, mask)
            if self.channel_budget > 0:
                x = channel_process(x, vector)
            x = conv1x1_mask(self.conv_pw_2, x, mask)
            x = bn_relu_mask(self.bn2, None, x, mask)
            meta["flops"] = torch.cat((meta["flops"], self.get_flops(vector, mask.hard).unsqueeze(dim=0)))
            x = apply_mask(x, mask)
            if self.before_residual:
                meta["saliency_mask"] = x

        meta["flops_full"] = torch.cat((meta["flops_full"], self.flops_full.repeat(x.shape[0]).unsqueeze(dim=0)))
        return (x, meta)

    def forward(self, inp):
        x, meta = inp
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.forward_block(inp)
        meta["stage_id"], meta["block_id"] = meta["stage_id"] + 1, meta["block_id"] + 1
        if self.final_activation == "linear":
            return_value = (identity + out[0], out[1]) if self.use_res_connect else out
            meta["saliency_mask"] = return_value[0]
            return return_value
        elif self.final_activation == "relu":
            return_value = (self.activation(identity + out[0]), out[1]) if self.use_res_connect \
                else (self.activation(out[0]), out[1])
            meta["saliency_mask"] = return_value[0]
            return return_value
        else:
            raise NotImplementedError

    def get_masked_feature(self, x, mask=None):
        if mask is None:
            return x
        else:
            return mask.float().expand_as(x) * x
