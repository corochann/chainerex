import os
import json
import numpy


class JSONEncoderEX(json.JSONEncoder):
    """Ref: https://stackoverflow.com/questions/27050108/convert-numpy-type-to-python"""
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super(JSONEncoderEX, self).default(obj)


def save_json(filepath, params):
    """save params in json format.

    Args:
        filepath (str): filepath to save args
        params (dict): args to be saved 

    """
    with open(filepath, 'w') as f:
        json.dump(params, f, indent=4, cls=JSONEncoderEX)


def load_json(filepath):
    """load params, whicch is stored in json format.

    Args:
        filepath (str): filepath to save args

    Returns (dict): params

    """
    with open(filepath, 'r') as f:
        params = json.load(f)
    return params

if __name__ == '__main__':
    import chainerex.utils as cl
    # Demo
    # Please remove 'result' directory, which contains 'args' file, after demo.
    args = {
        'a_int': 1,
        'b_str': 'string',
        'c_list': [1, 2, 3],
        'd_tuple': (1, 2),
        'n_int_scalar': numpy.array(1),
        'n_int_array': numpy.array([1]),
        'n_float': numpy.array([[1.0, 2.0], [3.0, 4.0]]),
    }
    out_dir = cl.create_timedir()
    filepath = os.path.join(out_dir, 'args')
    cl.save_json(filepath, args)

    # # Add additional information, it also work.
    # args.update({'e': 'ext_info'})
    # cl.save_json(os.path.join(out_dir, 'args'), args)
    load_args = load_json(filepath)
    print(type(load_args), load_args)
