
import math
from models.resnet_util import *


class ResNet_32x32(nn.Module):
    def __init__(self, layers, num_classes=10, pretrained=False, sparse=False, **kwargs):
        super(ResNet_32x32, self).__init__()

        assert len(layers) == 3
        block = BasicBlock
        self.sparse = sparse

        self.inplanes = 16
        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0], **kwargs)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2, **kwargs)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2, **kwargs)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def _make_layer(self, block, planes, blocks, stride=1, **kwargs):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, sparse=self.sparse, **kwargs))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, sparse=self.sparse, **kwargs))

        return nn.Sequential(*layers)

    def refresh_layer_id(self, meta):
        meta["stage_id"] += 1
        meta["block_id"] = 0
        return meta

    def forward(self, x, meta=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        meta["stage_id"], meta["block_id"], meta["masked_feat"] = 0, 0, None
        x, meta = self.layer1((x, meta))
        self.refresh_layer_id(meta)
        x, meta = self.layer2((x, meta))
        self.refresh_layer_id(meta)
        x, meta = self.layer3((x, meta))
        self.refresh_layer_id(meta)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, meta


class ResNet_BN_32x32(nn.Module):
    def __init__(self, layers, num_classes=10, pretrained=False, sparse=False, inp_h=32, inp_w=32,  **kwargs):
        super(ResNet_BN_32x32, self).__init__()

        if pretrained is not False:
            raise NotImplementedError('No pretrained models for 32x32 implemented')

        assert len(layers) == 3
        block = Bottleneck
        self.sparse = sparse

        self.inplanes = 16
        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        h, w = conv2d_out_dim(inp_h, kernel_size=3, stride=1, padding=1), \
               conv2d_out_dim(inp_w, kernel_size=3, stride=1, padding=1)
        flops_conv1 = 3 * 3 * h * w * 3 * self.inplanes
        self.layer1, h, w = self._make_layer(block, 16, h, w, layers[0], **kwargs)
        self.layer2, h, w = self._make_layer(block, 32, h, w, layers[1], stride=2, **kwargs)
        self.layer3, h, w = self._make_layer(block, 64, h, w, layers[2], stride=2, **kwargs)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        flops_fc = 64 * block.expansion * num_classes
        self.flops_basic = torch.Tensor([flops_conv1 + flops_fc])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, inp_h, inp_w, blocks, stride=1, **kwargs):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, inp_h, inp_w, stride, downsample, mask_block=True, sparse=self.sparse, **kwargs))
        self.inplanes = planes * block.expansion
        out_h = conv2d_out_dim(inp_h, kernel_size=1, stride=stride, padding=0)
        out_w = conv2d_out_dim(inp_w, kernel_size=1, stride=stride, padding=0)

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, out_h, out_w, sparse=self.sparse, mask_block=False, **kwargs))

        return nn.Sequential(*layers), out_h, out_w

    def refresh_layer_id(self, meta):
        meta["stage_id"] += 1
        meta["block_id"] = 0
        return meta

    def forward(self, x, meta=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        meta["stage_id"], meta["block_id"], meta["masked_feat"] = 0, 0, None
        meta["flops"], meta["flops_full"] = self.flops_basic.repeat(x.shape[0]).unsqueeze(dim=0).cuda(), \
                                            self.flops_basic.repeat(x.shape[0]).unsqueeze(dim=0).cuda()
        x, meta = self.layer1((x, meta))
        self.refresh_layer_id(meta)
        x, meta = self.layer2((x, meta))
        self.refresh_layer_id(meta)
        x, meta = self.layer3((x, meta))
        self.refresh_layer_id(meta)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, meta


def resnet8(sparse=False, **kwargs):
    return ResNet_32x32([1,1,1], sparse=sparse, **kwargs)

def resnet14(sparse=False, **kwargs):
    return ResNet_32x32([2,2,2], sparse=sparse, **kwargs)

def resnet20(sparse=False, **kwargs):
    return ResNet_32x32([3,3,3], sparse=sparse, **kwargs)

def resnet26(sparse=False, **kwargs):
    return ResNet_32x32([4,4,4], sparse=sparse, **kwargs)

def resnet32(sparse=False, **kwargs):
    return ResNet_32x32([5,5,5], sparse=sparse, **kwargs)

def resnet32_BN(sparse=False, **kwargs):
    return ResNet_BN_32x32([5,5,5], sparse=sparse, **kwargs)
