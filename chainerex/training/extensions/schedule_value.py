import os
import numpy

import chainer
from chainer.training import Trainer


def schedule_optimizer_value(epoch_list, value_list, optimizer_name='main',
                             attr_name='__auto'):
    """Set optimizer's hyperparameter according to value_list, scheduled on epoch_list. 
    
    Example usage:
    trainer.extend(schedule_optimizer_value([2, 4, 7], [0.008, 0.006, 0.002]))
    
    or 
    trainer.extend(schedule_optimizer_value(2, 0.008))

    Args:
        epoch_list (list, int or float): list of int. epoch to invoke this extension. 
        value_list (list, int or float): list of float. value to be set.
        optimizer_name: optimizer's name on trainer
        attr_name: attr name of optimizer to change value.
           if '__auto' is set, it will automatically infer learning rate attr name. 

    Returns (callable): extension function

    """
    if isinstance(epoch_list, list):
        if len(epoch_list) != len(value_list):
            raise ValueError('epoch_list length {} and value_list length {} '
                             'must be same!'
                             .format(len(epoch_list), len(value_list)))
    else:
        assert isinstance(epoch_list, float) or isinstance(epoch_list, int)
        assert isinstance(value_list, float) or isinstance(value_list, int)
        epoch_list = [epoch_list, ]
        value_list = [value_list, ]


    trigger = chainer.training.triggers.ManualScheduleTrigger(epoch_list,
                                                              'epoch')
    count = 0
    _attr_name = attr_name

    @chainer.training.extension.make_extension(trigger=trigger)
    def set_value(trainer: Trainer):
        nonlocal count, _attr_name
        value = value_list[count]
        optimizer = trainer.updater.get_optimizer(optimizer_name)

        # Infer attr name
        if count == 0 and _attr_name == '__auto':
            if isinstance(optimizer, chainer.optimizers.Adam):
                _attr_name = 'alpha'
            else:
                _attr_name = 'lr'

        print('updating {} to {}'.format(_attr_name, value))
        setattr(optimizer, _attr_name, value)
        count += 1

    return set_value


def schedule_target_value(epoch_list, value_list, target, attr_name):
    """Set optimizer's hyperparameter according to value_list, scheduled on epoch_list. 

    target is None -> use main optimizer

    Example usage:
    trainer.extend(schedule_target_value([2, 4, 7], [0.008, 0.006, 0.002], iterator, 'batch_size'))
    """
    if isinstance(epoch_list, list):
        if not isinstance(value_list, list):
            assert isinstance(value_list, float) or isinstance(value_list, int)
            value_list = [value_list, ]
        if len(epoch_list) != len(value_list):
            raise ValueError('epoch_list length {} and value_list length {} '
                             'must be same!'
                             .format(len(epoch_list), len(value_list)))
    else:
        assert isinstance(epoch_list, float) or isinstance(epoch_list, int)
        assert isinstance(value_list, float) or isinstance(value_list, int)
        epoch_list = [epoch_list, ]
        value_list = [value_list, ]

    trigger = chainer.training.triggers.ManualScheduleTrigger(epoch_list,
                                                              'epoch')
    count = 0

    @chainer.training.extension.make_extension(trigger=trigger)
    def set_value(trainer: Trainer):
        nonlocal count
        value = value_list[count]

        print('updating {} to {}'.format(attr_name, value))
        setattr(target, attr_name, value)
        count += 1

    return set_value
