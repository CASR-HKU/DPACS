
def conv1x1_mask(conv_module, x, mask):
    w = conv_module.weight.data
    mask.flops_per_position += w.shape[0]*w.shape[1]
    conv_module.__mask__ = mask
    return conv_module(x)


def conv3x3_dw_mask(conv_module, x, mask_dilate, mask):
    w = conv_module.weight.data
    mask.flops_per_position += w.shape[0]*w.shape[1]*w.shape[2]*w.shape[3]
    conv_module.__mask__ = mask
    return conv_module(x)


def conv3x3_mask(conv_module, x, mask_dilate, mask):
    w = conv_module.weight.data
    mask.flops_per_position += w.shape[0]*w.shape[1]*w.shape[2]*w.shape[3]
    conv_module.__mask__ = mask
    return conv_module(x)


def bn_relu_mask(bn_module, relu_module, x, mask, fast=False):
    bn_module.__mask__ = mask
    if relu_module is not None:
        relu_module.__mask__ = mask

    x = bn_module(x)
    x = relu_module(x) if relu_module is not None else x
    return x

