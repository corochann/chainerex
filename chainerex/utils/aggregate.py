import os
from sys import stderr

import pandas

from chainerex.utils.log import load_json
from chainerex.utils.filesys import collect_files


def aggregate_json(dirpath, file_name='args', filter_name='',
                     filepath_column='dirpath'):
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
            params.update({filepath_column: os.path.dirname(path)})
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


def aggregate_log(dirpath, file_name='log', filter_name='',
                    filepath_column='dirpath',
                    method='last', key=None):
    """aggregate LogReport format files under `dirpath` recursively.

    All Files with name `fila_name` under `dirpath` which contains
    `filter_name` is aggregated to one DataFrame.

    Args:
        dirpath (str): Root directory path.
        file_name (str): file name to aggregate
        filter_name (str or list): 
        filepath_column (str): filepath column name
        method (str): `last` is to take last reported value.
                      `minimum` is used to take minimum value at `key`.
        key (): 

    Returns (pandas.DataFrame): data frame which contains args info.

    """
    series_list = []
    error_count = 0
    for path in collect_files(dirpath, file_name, filter_name=filter_name):
        try:
            series = None
            params = load_json(path)
            log_df = pandas.DataFrame(params)

            if method == 'last':
                series = log_df.iloc[-1, :]
                #return last_series
            elif method == 'min':
                if key is not None:
                    series = log_df.ilox[log_df[key].idxmin(), :]

                    #return min_series
            if series is not None:
                series[filepath_column] = os.path.dirname(path)
                series_list.append(series)
                print('[DEBUG] series ', series)
            # params.update({filepath_column: path})
            # print('path', path, 'params', params)

        except Exception as e:
            print('[WARNING] load_json failed in aggregate_json with path {}'
                  .format(path))
            print(e)
            error_count += 1

    df = pandas.DataFrame(series_list)
    if error_count > 0:
        print('aggregate_json finished with error_count {}'.format(error_count))
    return df.reset_index()


def merge_json_and_log(dirpath, json_file_name='args', log_file_name='log',
                          filter_name='', filepath_column='filepath',
                          method='last', key=None):

    df_json = aggregate_json(dirpath, file_name=json_file_name,
                             filter_name=filter_name, filepath_column=filepath_column)
    df_log = aggregate_log(dirpath, file_name=log_file_name,
                           filter_name=filter_name,
                           filepath_column=filepath_column,
                           method=method, key=key)
    # TODO: Review which is better? how='inner' or 'outer'
    df = df_json.merge(df_log, how='outer')
    return df


if __name__ == '__main__':
    # df = aggregate_json('.', 'args', filter_name='')

    # df = aggregate_log('./results', 'log', filter_name='')

    df = merge_json_and_log('./results')
    print('df', df.shape, type(df))
    print(df)
