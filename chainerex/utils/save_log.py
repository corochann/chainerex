import json
import os


def save_json(filepath, params):
    """save params in json format.

    Args:
        filepath (str): filepath to save args
        params (dict): args to be saved 

    """
    with open(filepath, 'w') as f:
        json.dump(params, f, indent=4)

if __name__ == '__main__':
    import chainerex.utils as cl
    # Demo
    # Please remove 'result' directory, which contains 'args' file, after demo.
    args = {
        'a_int': 1,
        'b_str': 'string',
        'c_list': [1, 2, 3],
        'd_tuple': (1, 2),
    }
    out_dir = cl.create_timedir()
    cl.save_json(os.path.join(out_dir, 'args'), args)

    # # Add additional information, it also work.
    # args.update({'e': 'ext_info'})
    # cl.save_json(os.path.join(out_dir, 'args'), args)
