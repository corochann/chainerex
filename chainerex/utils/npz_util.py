import numpy


def get_npz_keys(filename):
    """Get keys of .npz file
    
    It can be also used to check chainer model's key.
    
    .. admonition:: Example

        >>> keys = get_npz_keys('hoge.npz')
        >>> print(keys)
    
    """
    with numpy.load(filename) as f:
        # import IPython; IPython.embed()
        # print('filename {} has following keys'.format(filename))
        # print(f.keys())
        keys = f.keys()
        return keys
