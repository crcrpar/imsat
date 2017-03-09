from __future__ import division

import numpy

from chainer.dataset import iterator


class MultiDatasetIterator(iterator.Iterator):

    _last_signal = object()

    def __init__(self, dataset1, dataset2, batch_size, repeat=True, shuffle=True,
                 n_processes=None, n_prefetch=1, shared_mem=None):
        assert len(dataset1) == len(dataset2)
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.batch_size = batch_size
        self._repeat = repeat
        if shuffle:
            self._order = numpy.random.permutation(len(dataset1))
        else:
            self._order = None

        self.current_position = 0
        self.epoch = 0
        self.is_new_epoch = False

    def __next__(self):
        if not self._repeat and self.epoch > 0:
            raise StopIteration

        i = self.current_position
        i_end = i + self.batch_size
        N = len(self.dataset1)

        if self._order is None:
            batch1 = self.dataset1[i:i_end]
            batch2 = self.dataset2[i:i_end]
        else:
            batch1 = [self.dataset1[index] for index in self._order[i:i_end]]
            batch2 = [self.dataset2[index] for index in self._order[i:i_end]]

        if i_end >= N:
            if self._repeat:
                rest = i_end - N
                if self._order is not None:
                    numpy.random.shuffle(self._order)
                if rest > 0:
                    if self._order is None:
                        batch1.extend(self.dataset1[:rest])
                        batch2.extend(self.dataset2[:rest])
                    else:
                        batch1.extend([self.dataset1[index]
                                       for index in self._order[:rest]])
                        batch2.extend([self.dataset2[index]
                                       for index in self._order[:rest]])
                self.current_position = rest
            else:
                self.current_position = N

            self.epoch += 1
            self.is_new_epoch = True

        else:
            self.is_new_epoch = False
            self.current_position = i_end

        return (batch1, batch2)

    next = __next__

    @property
    def epoch_detail(self):
        return self.epoch + self.current_position / len(self.dataset1)

    def serialize(self, serializer):
        self.current_position = serializer('current_position',
                                           self.current_position)
        self.epoch = serializer('epoch', self.epoch)
        self.is_new_epoch = serializer('is_new_epoch', self.is_new_epoch)
        if self._order is not None:
            serializer('_order', self._order)
