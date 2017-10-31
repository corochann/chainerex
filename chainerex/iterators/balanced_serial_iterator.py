from __future__ import division

import numpy

from chainer.dataset import iterator


class IndexIterator(iterator.Iterator):
    """
    
    IndexIterator is used internally in `BalancedSerialIterator`, as each 
    label's index iterator 
    """

    def __init__(self, index_list, shuffle=True, num=0):
        self.index_list = numpy.asarray(index_list)
        assert self.index_list.ndim == 1
        self.index_length = len(index_list)
        self.current_index_list = None
        self.current_pos = 0
        self.shuffle = shuffle
        self.num = num

        self.update_current_index_list()

    def update_current_index_list(self):
        if self.shuffle:
            self.current_index_list = numpy.random.permutation(self.index_list)
        else:
            self.current_index_list = self.index_list

    def __next__(self):
        return self.get_next_indices(self.num)

    def get_next_indices(self, num):
        """get next indices

        Args:
            num (int): number for indeices to extract.

        Returns (numpy.ndarray): 1d ndarray of indeces

        .. admonition:: Example

           >>> ii = IndexIterator([1, 3, 5, 10], shuffle=True)
           >>> print(ii.get_next_indices(5))
           [ 5  1 10  3 10]
           >>> print(ii.get_next_indices(5))
           [ 3  1  5 10  1]

        """

        indices = []
        if self.current_pos + num < self.index_length:
            indices.append(self.current_index_list[self.current_pos: self.current_pos + num])
            self.current_pos += num
        else:
            indices.append(self.current_index_list[self.current_pos:])
            num -= (self.index_length - self.current_pos)
            q, r = divmod(num, self.index_length)
            # for _ in range(q):
                # self.update_current_index_list()
                # indices.append(self.current_index_list)
            indices.append(numpy.tile(self.index_list, q))
            self.update_current_index_list()
            indices.append(self.current_index_list[:r])
            self.current_pos = r

        return numpy.concatenate(indices).ravel()

    def serialize(self, serializer):
        self.current_index_list = serializer('current_index_list',
                                             self.current_index_list)
        self.current_pos = serializer('current_pos', self.current_pos)


class BalancedSerialIterator(iterator.Iterator):

    """Dataset iterator that serially reads the examples with balancing label.

    This is a simple implementation of :class:`~chainer.dataset.Iterator`
    that just visits each example in either the order of indexes or a shuffled
    order.

    To avoid unintentional performance degradation, the ``shuffle`` option is
    set to ``True`` by default. For validation, it is better to set it to
    ``False`` when the underlying dataset supports fast slicing. If the
    order of examples has an important meaning and the updater depends on the
    original order, this option should be set to ``False``.

    This iterator saves ``-1`` instead of ``None`` in snapshots since some
    serializers do not support ``None``.

    Args:
        dataset: Dataset to iterate.
        batch_size (int): Number of examples within each batch.
        repeat (bool): If ``True``, it infinitely loops over the dataset.
            Otherwise, it stops iteration at the end of the first epoch.
        shuffle (bool): If ``True``, the order of examples is shuffled at the
            beginning of each epoch. Otherwise, examples are extracted in the
            order of indexes.

    """

    def __init__(self, dataset, batch_size, labels, repeat=True, shuffle=True,
                 batch_balancing=False, ignore_labels=None):
        assert len(dataset) == len(labels)
        labels = numpy.asarray(labels)
        if len(dataset) != labels.size:
            raise ValueError('dataset length {} and labels size {} must be '
                             'same!'.format(len(dataset), labels.size))
        labels = numpy.ravel(labels)
        # if labels.ndim != 1:
        #     raise ValueError('labels must be 1 dim, but got {} dim array'
        #                      .format(labels.ndim))
        self.dataset = dataset
        self.batch_size = batch_size
        self.labels = labels

        if ignore_labels is None:
            ignore_labels = []
        elif isinstance(ignore_labels, int):
            ignore_labels = [ignore_labels, ]
        self.ignore_labels = list(ignore_labels)
        self._repeat = repeat
        self._shuffle = shuffle
        self._batch_balancing = batch_balancing

        self.labels_iterator_dict = {}

        max_label_count = -1
        include_label_count = 0
        for label in numpy.unique(labels):
            label_index = numpy.argwhere(labels == label).ravel()
            # print('label', label, )
            label_count = len(label_index)
            ii = IndexIterator(label_index, shuffle=shuffle)
            self.labels_iterator_dict[label] = ii
            if label in self.ignore_labels:
                continue
            # --- below only for included labels ---
            if max_label_count < label_count:
                max_label_count = label_count
            include_label_count += 1

        self.max_label_count = max_label_count
        self.N_augmented = max_label_count * include_label_count
        self.reset()

    def __next__(self):
        if not self._repeat and self.epoch > 0:
            raise StopIteration

        self._previous_epoch_detail = self.epoch_detail

        i = self.current_position
        i_end = i + self.batch_size
        N = self.N_augmented

        batch = [self.dataset[index] for index in self._order[i:i_end]]

        if i_end >= N:
            if self._repeat:
                rest = i_end - N
                self._update_order()
                # if self._order is not None:
                #     numpy.random.shuffle(self._order)
                if rest > 0:
                    # if self._order is None:
                    #     batch.extend(self.dataset[:rest])
                    # else:
                    batch.extend([self.dataset[index]
                                  for index in self._order[:rest]])
                self.current_position = rest
            else:
                self.current_position = 0

            self.epoch += 1
            self.is_new_epoch = True
        else:
            self.is_new_epoch = False
            self.current_position = i_end

        return batch

    next = __next__

    @property
    def epoch_detail(self):
        return self.epoch + self.current_position / self.N_augmented

    @property
    def previous_epoch_detail(self):
        if self._previous_epoch_detail < 0:
            return None
        return self._previous_epoch_detail

    def serialize(self, serializer):
        self.current_position = serializer('current_position',
                                           self.current_position)
        self.epoch = serializer('epoch', self.epoch)
        self.is_new_epoch = serializer('is_new_epoch', self.is_new_epoch)
        if self._order is not None:
            try:
                serializer('order', self._order)
            except KeyError:
                serializer('_order', self._order)
        try:
            self._previous_epoch_detail = serializer(
                'previous_epoch_detail', self._previous_epoch_detail)
        except KeyError:
            # guess previous_epoch_detail for older version
            self._previous_epoch_detail = self.epoch + \
                (self.current_position - self.batch_size) / self.N_augmented
            if self.epoch_detail > 0:
                self._previous_epoch_detail = max(
                    self._previous_epoch_detail, 0.)
            else:
                self._previous_epoch_detail = -1.

        for label, index_iterator in self.labels_iterator_dict.items():
            self.labels_iterator_dict[label] = serializer(
                'index_iterator_{}'.format(label),
                self.labels_iterator_dict[label]
            )

    def _update_order(self):
        indices_list = []
        for label, index_iterator in self.labels_iterator_dict.items():
            if label in self.ignore_labels:
                # Not include index of ignore_labels
                continue
            indices_list.append(index_iterator.get_next_indices(
                self.max_label_count))
            # print('key, value = ', key, value)

        indices = numpy.concatenate(indices_list).ravel()
        self._order = numpy.random.permutation(indices)

    def reset(self):
        self._update_order()
        self.current_position = 0
        self.epoch = 0
        self.is_new_epoch = False

        # use -1 instead of None internally.
        self._previous_epoch_detail = -1.

    def show_label_stats(self):
        print('   label    count     rate     status')
        total = 0
        for label, index_iterator in self.labels_iterator_dict.items():
            count = len(index_iterator.index_list)
            total += count

        for label, index_iterator in self.labels_iterator_dict.items():
            count = len(index_iterator.index_list)
            rate = count / len(self.dataset)
            status = 'ignored' if label in self.ignore_labels else 'included'
            print('{:>8} {:>8} {:>8.4f} {:>10}'
                  .format(label, count, rate, status))


if __name__ == '__main__':
    x = numpy.arange(100) + 1
    # x = numpy.arange(10) + 1
    t = numpy.log10(x).astype(numpy.int32)
    print('x', x, 't', t)

    # print(numpy.tile(x, 0))
    # print(numpy.tile(x, 1))
    # print(numpy.tile(x, 2))
    import chainerex
    from chainer.datasets import TupleDataset

    # iterator = BalancedSerialIterator(TupleDataset(x, t), 16, labels=t)
    iterator = BalancedSerialIterator(TupleDataset(x, t), 16, labels=t, ignore_labels=1)
    print(iterator.N_augmented)
    a = iterator.next()
    print(a)
    a = iterator.next()
    print(a)
    iterator.show_label_stats()
