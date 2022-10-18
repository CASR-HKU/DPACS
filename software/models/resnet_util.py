try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
from .maskunit import *
from .layers import *
from models.channel_saliency import conv_forward, channel_process, ChannelVectorUnit
from models.utils import conv2d_out_dim, apply_mask, NoneMask, get_input_active_matrix
from torch.nn import functional as F
import torch


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    """Standard residual block """
    expansion = 1

    def __init__(self, inplanes, planes, inp_h, inp_w, stride=1, downsample=None, groups=1, base_width=64, dilation=1,
                 norm_layer=None, resolution_mask=False, mask_block=False,  channel_budget=-1, channel_unit_type="fc", 
                 group_size=1, budget=-1, before_residual=False, device="cuda:0", input_mask=False, **kwargs):
        super(BasicBlock, self).__init__()
        assert groups == 1
        assert dilation == 1

        self.input_mask = input_mask
        self.sparse = budget > 0
        self.device = device
        self.inp_height, self.inp_width = inp_h, inp_w
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.resolution_mask = resolution_mask
        self.mask_block = mask_block
        self.mask_sampler = nn.MaxPool2d(kernel_size=2)
        self.channel_budget = channel_budget
        self.spatial_budget = budget
        self.before_residual = before_residual
        self.input_mask_matrix = get_input_active_matrix(stride, inp_h, inp_w)
        self.flops_weight3x3 = torch.ones(1, 1, 3, 3).to(device)

        print(f'Bottleneck - sparse: {self.sparse}: inp {inplanes}, ' +
              f'oup {planes * self.expansion}, stride {stride}')

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        self.outp_height = conv2d_out_dim(inp_h, kernel_size=1, stride=stride, padding=0)
        self.outp_width = conv2d_out_dim(inp_w, kernel_size=1, stride=stride, padding=0)
        flops_conv1_full = torch.Tensor([9 * self.outp_height * self.outp_width * planes * inplanes])
        flops_conv2_full = torch.Tensor([9 * self.outp_height * self.outp_width * planes * planes])
        self.flops_downsample = torch.Tensor([self.outp_height*self.outp_width*planes*inplanes]) \
            if self.downsample is not None else torch.Tensor([0])
        self.flops_full = (flops_conv1_full + flops_conv2_full + self.flops_downsample).to(device=device)

        if self.sparse:
            if channel_unit_type == "fc":
                self.saliency = ChannelVectorUnit(in_channels=inplanes, out_channels=planes, device=device,
                                                  group_size=group_size, channel_budget=channel_budget, **kwargs)
            else:
                raise NotImplementedError

            if resolution_mask and not self.mask_block:
                pass
            else:
                self.masker = MaskUnit(channels=inplanes, out_height=self.inp_height,
                                               out_width=self.inp_width, stride=1, dilate_stride=1,
                                               budget=self.spatial_budget, input_resolution=True, **kwargs)
        else:
            self.mask = NoneMask()

    def forward(self, input):
        x, meta = input
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        if not self.sparse:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)
            out += identity
            meta["flops"] = torch.cat((meta["flops"], self.flops_full.repeat(x.shape[0]).unsqueeze(dim=0)))
        else:
            assert meta is not None
            meta["stride"] = self.stride
            # if self.channel_budget > 0:
            vector, meta = self.saliency(x, meta)
            conv_forward(self.conv1, None, None, vector, forward=False)
            conv_forward(self.conv2, None, vector, None, forward=False)

            m = self.obtain_mask(x, meta)
            mask_dilate, mask = m['dilate'], m['std']

            if self.input_mask:
                x = apply_mask(x, mask_dilate)
            x = conv3x3_mask(self.conv1, x, mask_dilate, mask)
            x = bn_relu_mask(self.bn1, self.relu, x, mask_dilate)
            x = apply_mask(x, mask)
            x = channel_process(x, vector)

            x = conv3x3_mask(self.conv2, x, mask_dilate, mask)
            x = bn_relu_mask(self.bn2, None, x, mask)
            out = identity + apply_mask(x, mask)
            meta["saliency_mask"] = x if self.before_residual else out
            meta["flops"] = torch.cat((meta["flops"], self.get_flops(vector, mask.hard, mask_dilate.hard).unsqueeze(dim=0)))
        meta["flops_full"] = torch.cat((meta["flops_full"], self.flops_full.repeat(x.shape[0]).unsqueeze(dim=0)))
        out = self.relu(out)
        meta["block_id"] += 1
        return out, meta

    def obtain_mask(self, x, meta):
        if self.resolution_mask and not self.mask_block:
            m = {"dilate": meta["masks"][-1]["std"], "std": meta["masks"][-1]["std"]}
        else:
            m = self.masker(x, meta)
        return m

    def get_flops(self, mask_c, mask_s, mask_s_up):
        if not self.sparse:
            return self.flops_full
        channel_sum = mask_c.sum(1)
        spatial_sum = mask_s.sum((1,2,3))

        flops_conv1 = (F.conv2d(mask_s_up, self.flops_weight3x3, stride=self.conv1.stride[0], padding=self.conv1.padding[0])).sum((1,2,3)) * channel_sum * self.inplanes
        flops_conv2 = (F.conv2d(mask_s, self.flops_weight3x3, stride=self.conv2.stride[0], padding=self.conv2.padding[0])).sum((1,2,3)) * channel_sum * self.planes
        # flops_conv1 = 9 * spatial_sum * channel_sum * self.inplanes
        # flops_conv2 = 9 * spatial_sum * channel_sum * self.planes
        spatial_flops = torch.tensor(0) if self.resolution_mask and not self.mask_block else self.masker.get_flops()
        channel_flops = torch.tensor(0) if not self.saliency.using else self.saliency.get_flops()
        flops_mask = spatial_flops + channel_flops
        # total
        flops = flops_conv1+flops_conv2+self.flops_downsample.to(flops_conv1.device)+flops_mask
        return flops.to(device=self.device)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, inp_h, inp_w, stride=1, downsample=None, groups=1, base_width=64, dilation=1,
                 norm_layer=None, resolution_mask=False, mask_block=False, mask_type="conv",
                 save_feat=False, conv1_act="relu", channel_budget=-1, channel_unit_type="fc",
                 group_size=1, dropout_ratio=0, dropout_stages=[-1], budget=-1, before_residual=False, device="cuda:0",
                 **kwargs):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.sparse = budget > 0
        print(f'Bottleneck - sparse: {self.sparse}: inp {inplanes}, hidden_dim {width}, ' +
              f'oup {planes * self.expansion}, stride {stride}')

        self.device = device
        # Both self.conv2 and self.downsample layers downsample the input when stride !=
        self.inplanes = inplanes
        self.planes = planes * self.expansion
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        if conv1_act == "relu":
            self.conv1_act = nn.ReLU(inplace=True)
        elif conv1_act == "leaky_relu":
            self.conv1_act = nn.LeakyReLU(inplace=True)
        elif conv1_act == "none":
            self.conv1_act = None
        else:
            raise NotImplementedError
        self.downsample = downsample
        self.stride = stride
        self.resolution_mask = resolution_mask
        self.mask_block = mask_block
        self.save_feat = save_feat
        self.mask_type = mask_type
        self.mask_sampler = nn.MaxPool2d(kernel_size=2)
        self.channel_budget = channel_budget
        self.spatial_budget = budget
        self.dropout_ratio = dropout_ratio
        self.dropout_stages = dropout_stages
        self.before_residual = before_residual
        self.flops_weight3x3 = torch.ones(1, 1, 3, 3).to(device)

        self.inp_height, self.inp_width = inp_h, inp_w
        self.outp_height = conv2d_out_dim(inp_h, kernel_size=1, stride=stride, padding=0)
        self.outp_width = conv2d_out_dim(inp_w, kernel_size=1, stride=stride, padding=0)
        flops_conv1_full = torch.Tensor([self.inp_height * self.inp_width * width * inplanes])
        flops_conv2_full = torch.Tensor([9 * self.outp_height * self.outp_width * width * width])
        flops_conv3_full = torch.Tensor([self.outp_height * self.outp_width * width * planes*self.expansion])
        self.flops_downsample = torch.Tensor([self.outp_height*self.outp_width*planes*self.expansion*inplanes]
                                            ) if self.downsample is not None else torch.Tensor([0])
        self.flops_full = (flops_conv1_full + flops_conv2_full + flops_conv3_full + self.flops_downsample).to(device=device)

        if self.sparse:
            if channel_unit_type == "fc":
                self.saliency = ChannelVectorUnit(in_channels=inplanes, out_channels=planes, device=device,
                                                  group_size=group_size, channel_budget=channel_budget, **kwargs)
            else:
                raise NotImplementedError

            if resolution_mask and not self.mask_block:
                pass
            else:
                self.masker =MaskUnit(channels=inplanes, out_height=self.inp_height,
                                               out_width=self.inp_width, stride=1, dilate_stride=1,
                                               budget=self.spatial_budget, input_resolution=True, **kwargs)
        else:
            self.mask = NoneMask()

    def forward_conv(self, x, conv1_mask=None):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if conv1_mask is not None:
            out = out * conv1_mask.unsqueeze(dim=1)
            if self.conv2.stride[0] == 2:
                conv1_mask = self.mask_sampler(conv1_mask)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        return out, conv1_mask

    def obtain_mask(self, x, meta):
        if self.resolution_mask and not self.mask_block:
            m = {"dilate": meta["masks"][-1]["std"], "std": meta["masks"][-1]["std"]}
        else:
            m = self.masker(x, meta)
        return m

    def forward(self, input):
        x, meta = input
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        if not self.sparse:
            if not isinstance(self.mask, NoneMask):
                mask = self.mask.process(x, 1,  meta["stage_id"])
                out, mask = self.forward_conv(x, mask)
                out = out * mask.unsqueeze(dim=1)
            else:
                out, _ = self.forward_conv(x)

            if self.save_feat:
                meta["feat_before"].append(out)
            out += identity
            if self.save_feat:
                meta["feat_after"].append(out)
            meta["flops"] = torch.cat((meta["flops"], self.flops_full.repeat(x.shape[0]).unsqueeze(dim=0)))
        else:
            assert meta is not None
            meta["stride"] = self.stride
            vector, meta = self.saliency(x, meta)
            conv_forward(self.conv1, None, None, vector, forward=False)
            conv_forward(self.conv2, None, vector, vector, forward=False)
            conv_forward(self.conv3, None, vector, None, forward=False)

            m = self.obtain_mask(x, meta)
            mask_dilate, mask = m['dilate'], m['std']

            x = conv1x1_mask(self.conv1, x, mask_dilate)
            x = bn_relu_mask(self.bn1, self.conv1_act, x, mask_dilate)
            x = apply_mask(x, mask_dilate)
            if self.channel_budget > 0:
                x = channel_process(x, vector)
            x = conv3x3_mask(self.conv2, x, mask_dilate, mask)
            x = bn_relu_mask(self.bn2, self.relu, x, mask)
            if self.channel_budget > 0:
                x = channel_process(x, vector)
            x = conv1x1_mask(self.conv3, x, mask)
            x = bn_relu_mask(self.bn3, None, x, mask)
            out = identity + apply_mask(x, mask)
            meta["saliency_mask"] = x if self.before_residual else out
            meta["flops"] = torch.cat((meta["flops"], self.get_flops(vector, mask_dilate.hard, mask.hard).unsqueeze(dim=0)))
        meta["flops_full"] = torch.cat((meta["flops_full"], self.flops_full.repeat(x.shape[0]).unsqueeze(dim=0)))
        meta["block_id"] += 1
        out = self.relu(out)
        return out, meta

    def get_flops(self, mask_c, mask_s_up, mask_s_down):
        if not self.sparse:
            return self.flops_full
        channel_sum = mask_c.sum(1)
        spatial_up_sum = mask_s_up.sum((1,2,3))
        spatial_down_sum = mask_s_down.sum((1,2,3))

        flops_conv1 = spatial_up_sum * channel_sum * self.inplanes
        # flops_conv2 = 9 * spatial_down_sum * channel_sum * channel_sum
        flops_conv2 = (F.conv2d(mask_s_up, self.flops_weight3x3, stride=self.conv2.stride[0],
                                padding=self.conv2.padding[0])).sum((1,2,3)) * channel_sum * channel_sum
        flops_conv3 = spatial_down_sum * self.planes * channel_sum
        spatial_flops = torch.tensor(0) if self.resolution_mask and not self.mask_block else self.masker.get_flops()
        channel_flops = torch.tensor(0) if not self.saliency.using else self.saliency.get_flops()
        flops_mask = spatial_flops + channel_flops
        # total
        flops = flops_conv1+flops_conv2+flops_conv3+self.flops_downsample.to(flops_conv1.device)+flops_mask

        return flops.to(device=self.device)


