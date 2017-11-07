"""
Stochastic Gradient Descent with warm Restarts (SGDR)

Cosine annling with restarting.
"""
from typing import Iterable

import chainer
from math import cos, pi

from chainer.training import Extension


class CosineAnnealing(Extension):

    """

    Args:
        epoch_list (list): Cosine annealing scheduling list
           If this is int or float, cosine annealing is scheduled without 
           restart and the value is considered as last epoch.
           `epoch_list` must be ascending order.
           It must contain begging and the end of epoch as well.
        lr_max (float): Starting learning rate, this is denoted as 
            `eta_max` in the paper. 
        lr_min (float): End learning rate, this is denoted as `eta_min` in
            the paper.
        optimizer_name (str): optimizer's name on trainer
        attr_name (str): attr name of optimizer to change value.
           if '__auto', it will automatically infer learning rate attr name 
        callback_fn (callable): This is called before restart.

    .. admonition:: Example

        >>> import os
        >>> from chainer import training, serializers
        >>> from chainer.training import extensions
        >>> trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
        >>> # --- observe_lr ---
        >>> trainer.extend(extensions.observe_lr(observation_key='lr'))
        >>> # --- Callback function for snapshot ensemble ---
        >>> def callback_fn(trainer, stage):
        >>>     print('callback_fn stage={}'.format(stage))
        >>>     model_path = os.path.join(out_dir, 'model_cosine{}.npz'.format(stage))
        >>>     print('save model to {}'.format(model_path))
        >>>     serializers.save_npz(model_path, model)
        
        >>> # --- CosineAnnealing of learning rate ---
        >>> trainer.extend(CosineAnnealing(epoch_list=[0, 2, 4, 8, args.epoch], 
        >>>     lr_max=0.1, lr_min=0.0, callback_fn=callback_fn verbose=False), 
        >>>     trigger=(100, 'iteration'))

    """

    def __init__(self, epoch_list, lr_max=0.1, lr_min=0.0,
                 optimizer_name='main', attr_name='__auto', verbose=True,
                 callback_fn=None):
        if not isinstance(epoch_list, Iterable):
            raise TypeError('epoch_list must be list, got {}'
                            .format(type(epoch_list)))
        self.epoch_list = epoch_list
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.current_stage = 0
        self.optimizer_name = optimizer_name
        self.attr_name = attr_name
        self.verbose = verbose
        self.optimizer = None
        self.callback_fn = None

    def __call__(self, trainer):
        current_epoch = trainer.updater.epoch_detail
        # if self.current_stage >= len(self.epoch_list) - 1:
        if current_epoch < self.epoch_list[0] or current_epoch > self.epoch_list[-1]:
            # Out of range of this scheduler, do nothing.
            return

        # --- Check current stage ---
        epoch_stage_start = self.epoch_list[self.current_stage]
        epoch_stage_end = self.epoch_list[self.current_stage + 1]
        # print(current_epoch, epoch_stage_start, epoch_stage_end)
        if epoch_stage_start <= current_epoch and current_epoch < epoch_stage_end:
            # `current_stage` is same with previous.
            pass
        else:
            # Need to update `current_stage`
            for i, epoch in enumerate(self.epoch_list):
                if epoch > current_epoch:
                    if self.verbose:
                        print('current_stage updated from {} to {}'
                              .format(self.current_stage, i - 1))

                    if self.callback_fn is not None:
                        # You may take snapshot for snapshot ensembling etc
                        self.callback_fn(trainer, self.current_stage)
                    self.current_stage = i - 1
                    epoch_stage_start = self.epoch_list[self.current_stage]
                    epoch_stage_end = self.epoch_list[self.current_stage + 1]
                    break
            if self.current_stage >= len(self.epoch_list) - 1:
                # It already reaches last epoch in epoch_list, do nothing
                return

        # Get optimizer
        self.optimizer = self.optimizer or trainer.updater.get_optimizer(self.optimizer_name)

        # Infer attr name (only once)
        if self.attr_name == '__auto':
            if isinstance(self.optimizer, chainer.optimizers.Adam):
                self.attr_name = 'alpha'
            else:
                self.attr_name = 'lr'

        # --- Main code: update learning rate ---
        t_cur = current_epoch - epoch_stage_start
        t_i = epoch_stage_end - epoch_stage_start
        value = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
            1 + cos(pi * t_cur / t_i))
        setattr(self.optimizer, self.attr_name, value)
        if self.verbose:
            print('updating {} to {}'.format(self.attr_name, value))
