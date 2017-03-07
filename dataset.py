from __future__ import print_function

import numpy
import chainer


class IMSATIterator(chainer.iterators.SerialIterator):

    def __init__(self, dataset, batch_size, repeat=True, shuffle=True):
        super(IMSATIterator, self).__init__(dataset, batch_size, repeat=repeat, shuffle=shuffle)
        self._N = len(self.dataset)

    def __next__(self):
        if not self._repeat and self.epoch > 0:
            raise StopIteration

        i = self.current_position
        i_end = i + self.batch_size
        N = len(self.dataset)

        if self._order is None:
            batch = self.dataset[i:i_end]
            index = list(range(i, i_end))
        else:
            batch = [self.dataset[index] for index in self._order[i:i_end]]
            index = self._order[i:i_end]

        if i_end >= N:
            if self._repeat:
                rest = i_end - N
                if self._order is not None:
                    numpy.random.shuffle(self._order)
                if rest > 0:
                    if self._order is None:
                        batch += list(self.dataset[:rest])
                        index += list(range(rest))
                    else:
                        batch += [self.dataset[index]
                                  for index in self._order[:rest]]
                        index += [index for index in self._order[:rest]]
                self.current_position = rest
            else:
                self.current_position = N

            self.epoch += 1
            self.is_new_epoch = True
        else:
            self.is_new_epoch = False
            self.current_position = i_end

        return (batch, index)

    next = __next__
