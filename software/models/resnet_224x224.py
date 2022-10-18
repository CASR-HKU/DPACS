
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
from models.resnet_util import *


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}




class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, model_cfg=None,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, inp_h=224, inp_w=224,
                 norm_layer=None, sparse=False, width_mult=1., device="cuda:0", **kwargs):
        super(ResNet, self).__init__()
        self.sparse = sparse

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = int(64*width_mult)
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        if "hardware" in model_cfg:
            h, w = conv2d_out_dim(inp_h, kernel_size=3, stride=2, padding=1), \
                           conv2d_out_dim(inp_w, kernel_size=3, stride=2, padding=1)
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=2, padding=1, bias=False)
        else:
            h, w = conv2d_out_dim(inp_h, kernel_size=7, stride=2, padding=3), \
                           conv2d_out_dim(inp_w, kernel_size=7, stride=2, padding=3)
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)

        self.flops_conv1 = 7 * 7 * h * w * self.inplanes * 3
        h = conv2d_out_dim(h, kernel_size=3, stride=2, padding=1)
        w = conv2d_out_dim(w, kernel_size=3, stride=2, padding=1)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        kwargs["device"] = device
        self.layer1, h, w = self._make_layer(block, int(64*width_mult), layers[0], inp_w=w, inp_h=h,
                                             model_cfg=model_cfg, current_stage=0, **kwargs)
        self.layer2, h, w = self._make_layer(block, int(128*width_mult), layers[1], stride=2, inp_w=w, inp_h=h,
                                       dilate=replace_stride_with_dilation[0], model_cfg=model_cfg, current_stage=1, **kwargs)
        self.layer3, h, w = self._make_layer(block, int(256*width_mult), layers[2], stride=2, inp_w=w, inp_h=h,
                                       dilate=replace_stride_with_dilation[1], model_cfg=model_cfg, current_stage=2, **kwargs)
        self.layer4, h, w = self._make_layer(block, int(512*width_mult), layers[3], stride=2, inp_w=w, inp_h=h,
                                       dilate=replace_stride_with_dilation[2], model_cfg=model_cfg, current_stage=3, **kwargs)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if model_cfg == "hardware":
            self.fc = nn.Linear(int(512*width_mult * 2), num_classes)
            self.flops_fc = int(512*width_mult * 2) * num_classes
        else:
            self.fc = nn.Linear(int(512*width_mult * block.expansion), num_classes)
            self.flops_fc = int(512*width_mult * block.expansion)

        self.flops_basic = torch.Tensor([self.flops_fc + self.flops_conv1]).to(device=device)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, inp_h, inp_w, stride=1, dilate=False, model_cfg="baseline",
                    current_stage=-1, **kwargs):
        norm_layer = self._norm_layer
        base_width = self.base_width
        if model_cfg == "hardware" and current_stage == 3:
            block.expansion = 2

        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, inp_h, inp_w, stride, downsample, self.groups, base_width,
                            previous_dilation, norm_layer, mask_block=True, **kwargs))
        self.inplanes = planes * block.expansion
        out_h = conv2d_out_dim(inp_h, kernel_size=1, stride=stride, padding=0)
        out_w = conv2d_out_dim(inp_w, kernel_size=1, stride=stride, padding=0)
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, out_h, out_w, groups=self.groups, base_width=base_width,
                                dilation=self.dilation, norm_layer=norm_layer, mask_block=False, **kwargs))
        return nn.Sequential(*layers), out_h, out_w

    def refresh_layer_id(self, meta):
        meta["stage_id"] += 1
        meta["block_id"] = 0
        return meta

    def forward(self, x, meta=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        meta["stage_id"], meta["block_id"], meta["saliency_mask"] = 0, 0, x
        meta["flops"], meta["flops_full"] = self.flops_basic.repeat(x.shape[0]).unsqueeze(dim=0), \
                                            self.flops_basic.repeat(x.shape[0]).unsqueeze(dim=0)
        x, meta = self.layer1((x,meta))
        self.refresh_layer_id(meta)
        x, meta = self.layer2((x,meta))
        self.refresh_layer_id(meta)
        x, meta = self.layer3((x,meta))
        self.refresh_layer_id(meta)
        x, meta = self.layer4((x,meta))

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x, meta

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model

def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>'_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>'_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    print('Model: Resnet 34')
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>'_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    print('Model: Resnet 50')
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>'_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    print('Model: Resnet 101')
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(cfg, pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>'_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(cfg, 'resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(cfg, pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet(cfg, 'resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(cfg, pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet(cfg, 'resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(cfg, pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet(cfg, 'wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(cfg, pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet(cfg, 'wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)
