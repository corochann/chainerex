"""
Use `cache_load_npz` if `preprocess_fn` return numpy array data. 
Use `cache_load_pandas_hdf5` if `preprocess_fn` return pandas Series/DataFrame.
"""
import os

import numpy
import pandas


def cache_load_npz(filepath, preprocess_fn, *args, **kwargs):
    """Load cached dataset if possible, otherwise create it by `preprocess_fn`

    When dataset is not created yet, it will create using `preprocess_fn`,
    and save it to filepath.
    From next time, it will simply load dataset from `filepath` thus no 
    `preprocess_fn` is never invoked.

    `preprocess_fn` is expected to return numpy data

    Args:
        filepath (str): It is recommended to have '.npz' extension.
        preprocess_fn (callable): It must return data, whose type is numpy
            It may be None, if `filepath` is guaranteed to exist
        *args: args for `preprocess_fn`
        **kwargs: kwargs for `preprocess_fn`

    Returns: numpy dataset

    """
    if not os.path.exists(filepath):
        if preprocess_fn is None:
            raise ValueError('filepath {} does not exist, '
                             'preprocess_fn must not be None'.format(filepath))
        # Preprocess and cache(save) datasets
        print('Preprocessing dataset...')
        datasets = preprocess_fn(*args, **kwargs)
        if not isinstance(datasets, tuple):
            datasets = (datasets, )
        numpy.savez(filepath, *datasets)

    # Now the datasets is ready.
    assert os.path.exists(filepath)
    load_data = numpy.load(filepath)
    result = []
    i = 0
    while True:
        key = 'arr_{}'.format(i)
        if key in load_data.keys():
            result.append(load_data[key])
            i += 1
        else:
            break
    if len(result) == 1:
        result = result[0]
    return result


def cache_load_pandas_hdf5(filepath, preprocess_fn, *args, **kwargs):
    """Load cached dataset if possible, otherwise create it by `preprocess_fn`

    When dataset is not created yet, it will create using `preprocess_fn`,
    and save it to filepath.
    From next time, it will simply load dataset from `filepath` thus no 
    `preprocess_fn` is never invoked.

    `preprocess_fn` is expected to return pandas Series or DataFrame

    Args:
        filepath (str): It is recommended to have '.h5' extension.
        preprocess_fn (callable): It must return data, whose type is either 
            Series/DataFrame. It may be None, if `filepath` exists.
        *args: args for `preprocess_fn`
        **kwargs: kwargs for `preprocess_fn`

    Returns: pandas Series/DataFrame dataset

    """
    if not os.path.exists(filepath):
        if preprocess_fn is None:
            raise ValueError('filepath {} does not exist, '
                             'preprocess_fn must not be None'.format(filepath))
        # Preprocess and cache(save) datasets
        print('Preprocessing dataset...')
        datasets = preprocess_fn(*args, **kwargs)
        if not isinstance(datasets, tuple):
            datasets = (datasets, )
        store = pandas.HDFStore(filepath)
        for i, d in enumerate(datasets):
            store['arr_{}'.format(i)] = d
        store.close()

    # Now the datasets is ready.
    assert os.path.exists(filepath)
    load_store = pandas.HDFStore(filepath)
    result = []
    i = 0
    while True:
        key = '/arr_{}'.format(i)
        if key in load_store.keys():
            result.append(load_store[key])
            i += 1
        else:
            break
    if len(result) == 1:
        result = result[0]
    load_store.close()
    return result


if __name__ == '__main__':
    # Demo
    # Please remove data.npz/data.h5 after demo

    # mode 1: cache_load_npz demo
    # mode 2: cache_load_pandas_hdf5 demo
    mode = 2

    if mode == 1:
        # Once you define preprocess_fn which prepares numpy data,
        # it can be automatically cached and loaded with `cache_load` method
        def preprocess_fn(scale):
            # This preprocess_fn return numpy arrays
            x = numpy.arange(10)
            t = x * scale
            return x, t
        x, t = cache_load_npz('data.npz', preprocess_fn, 3)
        print('x', x)
        print('t', t)
    elif mode == 2:
        # Once you define preprocess_fn which prepares numpy data,
        # it can be automatically cached and loaded with `cache_load` method
        def preprocess_fn(scale):
            # This preprocess_fn return DataFrame
            x = numpy.arange(10)
            t = x * x
            df1 = pandas.DataFrame({'x': x, 't': t})
            df2 = pandas.DataFrame({'x2': x*scale, 't2': t*scale})
            return df1, df2
        df1, df2 = cache_load_pandas_hdf5('data.h5', preprocess_fn, 3)
        print('df1', df1)
        print('df2', df2)
