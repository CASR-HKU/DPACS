import math
import torch.nn.functional as F
import torch


def apply_mask(x, mask):
    mask_hard = mask.hard
    assert mask_hard.shape[0] == x.shape[0]
    assert mask_hard.shape[2:4] == x.shape[2:4], (mask_hard.shape, x.shape)
    return mask_hard.float().expand_as(x) * x


def ponder_cost_map(masks):
    """ takes in the mask list and returns a 2D image of ponder cost """
    assert isinstance(masks, list)
    out = None
    for mask in masks:
        m = mask['std'].hard
        assert m.dim() == 4
        m = m[0]  # only show the first image of the batch
        if out is None:
            out = m
        else:
            out += F.interpolate(m.unsqueeze(0),
                                 size=(out.shape[1], out.shape[2]), mode='nearest').squeeze(0)
    return out.squeeze(0).cpu().numpy()


def conv2d_out_dim(dim, kernel_size, padding=0, stride=1, dilation=1, ceil_mode=False):
    if ceil_mode:
        return int(math.ceil((dim + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1))
    else:
        return int(math.floor((dim + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1))


class NoneMask:
    def __init__(self):
        pass

    def process(self, feat, stride, curr_block):
        bs, c, h, w = feat.shape
        if stride:
            return torch.ones((bs, 1, int(h/2), int(w/2))).cuda()
        return torch.ones(bs, 1, h, w).cuda()


def get_input_active_matrix(stride, h, w):
    if stride == 1:
        base_row = [4] + [9 for _ in range(w-2)] + [4]
        top_down_row = [4 for _ in range(w)]
        matrix = [top_down_row] + [base_row for _ in range(h-2)] + [top_down_row]
        return matrix
    elif stride == 2:
        matrix, odd_row_base, even_row_base = [], [], []
        for i in range(int(w/2)-1):
            odd_row_base.append(1)
            odd_row_base.append(2)
            even_row_base.append(2)
            even_row_base.append(4)
        for i in range(2):
            odd_row_base.append(1)
            even_row_base.append(2)

        for i in range(int(h/2)-1):
            matrix.append(odd_row_base)
            matrix.append(even_row_base)
        for i in range(2):
            matrix.append(odd_row_base)
    else:
        raise NotImplementedError
    return matrix


if __name__ == '__main__':
    m = get_input_active_mask(2,14,14)
    print(m)
