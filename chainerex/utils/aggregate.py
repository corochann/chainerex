import os
from sys import stderr

import pandas

from chainerex.utils.log import load_json
from chainerex.utils.filesys import collect_files


def aggregate_json(dirpath, file_name='args', filter_name=''):
    """aggregate files under `dirpath` recursively.
    
    All Files with name `fila_name` under `dirpath` which contains
    `filter_name` is aggregated to one DataFrame.

    Args:
        dirpath (str): Root directory path.
        filter_name (str or list): 

    Returns (pandas.DataFrame): data frame which contains args info.

    """
    df_dict = {}
    error_count = 0
    for index, path in enumerate(collect_files(dirpath, file_name)):
        try:
            params = load_json(path)
            params.update({'filepath': path})
            # print('path', path, 'params', params)
            df_dict.update({index: params})
            df_list.append(params)
        except Exception as e:
            print('[WARNING] load_json failed in aggregate_json with path {}'
                  .format(path))
            print(e)
            error_count += 1

    df = pandas.DataFrame(df_dict)
    if error_count > 0:
        print('aggregate_json finished with error_count {}'.format(error_count))
    # return df.transpose()
    #return df
    return df.transpose().reset_index()

if __name__ == '__main__':
    df = aggregate_json('.', 'args')
    print('df', df.shape, type(df))
    print(df)
