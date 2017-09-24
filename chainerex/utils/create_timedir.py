import os
from datetime import datetime as dt


def create_timedir(prefix='results/', tag=None, create_dir=True):
    tdatetime = dt.now()
    tstr = tdatetime.strftime('%Y%m%d-%H%M%S')
    if tag:
        tstr = tstr + '-{}'.format(tag)
    out_dir = prefix + tstr
    if create_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
        #print('[DEBUG] created {}'.format(out_dir))
    return out_dir

if __name__ == '__main__':
    # Demo
    create_timedir()
    # Please remove created `result` dir after demo.
