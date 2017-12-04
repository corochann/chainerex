from chainer import function_node
from chainer.utils import type_check


class PyramidAdd(function_node.FunctionNode):
    """
    
    Add different channel shaped array.
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
            # lhs.shape[1] >= rhs.shape[1],
            lhs.shape[2] == rhs.shape[2],
            lhs.shape[3] == rhs.shape[3],
        )

    def forward(self, x):
        if x[0].shape[1] >= x[1].shape[1]:
            lhs, rhs = x[:2]
        else:
            rhs, lhs = x[:2]
        self.rhs_ch = rhs.shape[1]
        if lhs.shape[1] > rhs.shape[1]:
            lhs[:, :self.rhs_ch, :, :] += rhs
        else:
            lhs += rhs
        return lhs,

    def backward(self, indexes, gy):
        return gy[0], gy[0][:, :self.rhs_ch, :, :]


def pyramid_add(lhs, rhs):
    return PyramidAdd().apply((lhs, rhs))[0]

# example usage
# h = pyramid_add(h, x)