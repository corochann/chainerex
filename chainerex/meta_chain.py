"""
Ref: https://www.slideshare.net/ssuser38b704/ll-lang-blackmagic?next_slideshow=1
"""
import os
import pickle
from types import MethodType, FunctionType, LambdaType

from chainer import Chain, serializers, Link


def _logging_hook(func):
    def __(self, *args, **kwargs):
        #print('before: {0}'.format(func.__name__))
        ret = func(self, *args, **kwargs)
        #print('args: ', args)
        #print('kwargs: ', kwargs)
        self._init_args = args
        self._init_kwargs = kwargs
        #print('after : {0}'.format(func.__name__))
        return ret

    return __


class MetaChain(type):

    def __new__(cls, name, bases, dict):
        #print('name : ', name)
        #print('bases: ', bases)
        #print('dict : ', dict)
        dict.update({
            '__init__': _logging_hook(dict['__init__']),
        })
        #setattr(cls, '_get_save_path', cls._get_save_path)
        bases += (MetaChainClass,)
        return type.__new__(cls, name, bases, dict)


class MetaChainClass(Chain):
    _init_args = None
    _init_kwargs = None

    @classmethod
    def _get_save_path(cls, dirpath):
        class_name = cls.__name__
        args_path = os.path.join(dirpath, class_name + '.pkl')
        model_path = os.path.join(dirpath, class_name + '.model')
        return args_path, model_path

    def save(self, dirpath):
        print('save to {}'.format(dirpath))
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)

        args_path, model_path = self._get_save_path(dirpath)
        # 1. Save init args
        init_dict = {
            'args': self._init_args,
            'kwargs': self._init_kwargs
        }
        with open(args_path, mode='wb') as f:
            pickle.dump(init_dict, f)

        # 2. Save the model parameters
        serializers.save_npz(model_path, self)

        print('model_path: {}'.format(model_path))
        print('init_args: ', self._init_args)
        print('init_kwargs: ', self._init_kwargs)
        pass

    @classmethod
    def load(cls, dirpath):
        print('load')
        args_path, model_path = cls._get_save_path(dirpath)

        # 1. Constract instance with init args/kwargs
        with open(args_path, mode='rb') as f:
            init_dict = pickle.load(f)

        print('init_dict', init_dict)
        model = cls(*init_dict['args'], **init_dict['kwargs'])
        serializers.load_npz(model_path, model)
        return model
