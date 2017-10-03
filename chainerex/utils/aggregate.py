import os
from sys import stderr

import pandas

from chainerex.utils.log import load_json
from chainerex.utils.filesys import collect_files


def aggregate_json(dirpath, file_name='args', filter_name='',
                     filepath_column='filepath'):
    """aggregate files under `dirpath` recursively.
    
    All Files with name `fila_name` under `dirpath` which contains
    `filter_name` is aggregated to one DataFrame.

    Args:
        dirpath (str): Root directory path.
        filter_name (str or list): 
        filepath_column (str): filepath column name

    Returns (pandas.DataFrame): data frame which contains args info.

    """
    df_list = []
    error_count = 0
    for path in collect_files(dirpath, file_name, filter_name=filter_name):
        try:
            params = load_json(path)
            params.update({filepath_column: path})
            # print('path', path, 'params', params)
            df_list.append(params)
        except Exception as e:
            print('[WARNING] load_json failed in aggregate_json with path {}'
                  .format(path))
            print(e)
            error_count += 1

    df = pandas.DataFrame(df_list)
    if error_count > 0:
        print('aggregate_json finished with error_count {}'.format(error_count))
    return df


if __name__ == '__main__':
    df = aggregate_json('.', 'args', filter_name='')
    print('df', df.shape, type(df))
    print(df)
