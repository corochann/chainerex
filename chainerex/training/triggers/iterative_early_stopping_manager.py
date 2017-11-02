from chainerex.training.triggers.early_stopping_trigger import EarlyStoppingTrigger


class IterativeEarlyStoppingManager(object):
    """
    
    Examples:
        from chainerex.training.triggers import IterativeEarlyStoppingManager

        iesm = IterativeEarlyStoppingManager()
        trainer = Trainer(updater, stop_trigger=iesm.stop_trigger)
        
        schedule_lr_list = [0.1, 0.001]
        def extension_fn(trainer):
            index = iesm.iterate_count
            optimizer.lr = schedule_lr_list[index]
        trainer.extend(extension_fn, trigger=iesm.extension_trigger)

    """

    def increment_iterate_count(self, trainer):
        self.iterate_count += 1
        if self.verbose:
            print('updating count to {}'.format(self.iterate_count))

    def __init__(self, max_iterate_count=-1,
                 trigger=(1, 'epoch'), monitor='main/loss', patients=3,
                 mode='auto', verbose=False, max_epoch=100, debug=False):
        self.extension_trigger = EarlyStoppingTrigger(
            trigger=trigger, monitor=monitor, patients=patients,
            mode=mode, verbose=verbose, max_epoch=max_epoch, debug=debug)
        self.extension_trigger.set_on_condition_listener(
            self.increment_iterate_count
        )
        self.verbose = verbose
        self.max_epoch = max_epoch
        self.stop_trigger = self.stop_condition
        self.max_iterate_count = max_iterate_count
        self.iterate_count = 0

    def stop_condition(self, trainer):
        # 1. Check epoch
        if self.max_epoch >=0 and trainer.updater.epoch_detail >= self.max_epoch:
            return True

        # 2. Check iterative count
        if self.max_iterate_count >=0 and self.max_iterate_count > self.iterate_count:
            return True
        return False

    @property
    def iterate_index(self):
        return self.iterate_count - 1
