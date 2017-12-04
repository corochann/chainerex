from chainer import function_node
from chainer.utils import type_check


# TODO: Test not done yet!
class PyramidAdd(function_node.FunctionNode):
    """
    
    Add different channel shaped array.

    This function is not commutable, lhs and rhs acts different!

    Add different channel shaped array.
    lhs is h, and rhs is x.
    h.shape[1] must be always equal or larger than x.shape[1]...
    output channel is always h.shape[1].
    
    x.shape[1] is smaller than h.shape[1], x is virtually padded with 0

    """

    def __init__(self):
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
            lhs.shape[1] >= rhs.shape[1],
            lhs.shape[2] == rhs.shape[2],
            lhs.shape[3] == rhs.shape[3],
        )

    def forward(self, x):
        lhs, rhs = x[:2]
        self.rhs_ch = rhs.shape[1]
        if lhs.shape[1] > rhs.shape[1]:
            lhs[:, :self.rhs_ch, :, :] += rhs
        else:
            lhs += rhs
        return lhs,

    def backward(self, indexes, gy):
        return gy[0], gy[0][:, :self.rhs_ch, :, :]


def pyramid_add(lhs, rhs):
    """
    
    # x: (mb, ch_x, h, w), h: (mb, ch_h, h, w)
    # output h: (mb, ch_h, h, w) regardless of the size of `ch_x`.
    h = pyramid_add(h, x)
    
    Args:
        lhs: 
        rhs: 

    Returns:

    """
    return PyramidAdd().apply((lhs, rhs))[0]

# example usage
# h = pyramid_add(h, x)