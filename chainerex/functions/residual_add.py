import chainer
from chainer import function_node
from chainer.utils import type_check


def reshape(x, channels):
    """Referenced from https://github.com/takedarts/resnetfamily"""
    if x.shape[1] < channels:
        xp = chainer.cuda.get_array_module(x)
        p = xp.zeros(
            (x.shape[0], channels - x.shape[1], x.shape[2], x.shape[3]),
            dtype=x.dtype)
        x = chainer.functions.concat((x, p), axis=1)
    elif x.shape[1] > channels:
        x = x[:, :channels, :]
    return x


class ResidualAdd(function_node.FunctionNode):
    """
    
    Be careful that this function is not commutable, lhs and rhs acts different!

    Add different channel shaped array.
    lhs is h, and rhs is x.
    output channel is always h.shape[1].
    
    If x.shape[1] is smaller than h.shape[1], x is virtually padded with 0
    If x.shape[1] is bigger than h.shape[1], only first h.shape[1] axis is used
        to add x to h.
    
    """

    def __init__(self):
        self.lhs_ch = None
        self.rhs_ch = None

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        lhs = in_types[0]
        rhs = in_types[1]
        type_check.expect(
            # lhs.dtype.kind == 'f',
            # rhs.dtype.kind == 'f',
            lhs.ndim == 4,
            rhs.ndim == 4,
            lhs.shape[0] == rhs.shape[0],
            # lhs.shape[1] >= rhs.shape[1],
            lhs.shape[2] == rhs.shape[2],
            lhs.shape[3] == rhs.shape[3],
        )

    def forward(self, x):
        lhs, rhs = x[:2]
        self.lhs_ch = lhs.shape[1]
        self.rhs_ch = rhs.shape[1]
        if self.lhs_ch < self.rhs_ch:
            lhs += rhs[:, :self.lhs_ch, :, :]
            return lhs,
            # pyramid add
            # rhs[:, :self.lhs_ch, :, :] += lhs
            # return rhs,

        elif self.lhs_ch > self.rhs_ch:
            lhs[:, :self.rhs_ch, :, :] += rhs
            return lhs,
        else:
            lhs += rhs
            return lhs,

    def backward(self, indexes, gy):
        if self.lhs_ch < self.rhs_ch:
            return gy[0], reshape(gy[0], self.rhs_ch)
            # pyramid add
            # return gy[0][:, :self.lhs_ch, :, :], gy[0]
        elif self.lhs_ch > self.rhs_ch:
            return gy[0], gy[0][:, :self.rhs_ch, :, :]
        else:
            return gy[0], gy[0]


def residual_add(lhs, rhs):
    """

    # x: (mb, ch_x, h, w), h: (mb, ch_h, h, w)
    # output h: (mb, ch_h, h, w). shape is always same with h (lhs),
    # regardless of the size of `ch_x`.
    h = pyramid_add(h, x)
    
    Args:
        lhs: 
        rhs: 

    Returns:

    """
    return ResidualAdd().apply((lhs, rhs))[0]

# example usage
# h = pyramid_add(h, x)