from __future__ import print_function

import numpy


class Data:

    def __init__(self, data, label):
        self._data = data
        self._label = label
        self._index = numpy.arange(len(self._data))

    def get_multi_data(self, idx_list):
        return self._data[idx_list]

    def get(self, n, _index=False):
        indices = numpy.random.permutation(self._N)
        if _index:
            return self._data[indices[:n]], self._label[indices[:n]], self._index[indices[:n]]
        else:
            return self._data[indices[:n]], self._label[indices[:n]]

    @property
    def data(self):
        return self._data

    @property
    def label(self):
        return self._label
