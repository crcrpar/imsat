from __future__ import print_function

import numpy


class DataGenerator:

    def __init__(self, data, label):
        self._data = data
        self._label = label
        self._index = numpy.arange(len(self._data))

    def get_index_data(self, index_list):
        return self._data[index_list]

    def get(self, n, need_index=False):
        indices = numpy.random.permutation(self._data.shape[0])
        if need_index:
            return self._data[indices[:n]], self._label[indices[:n]],  self._index[indices[:n]]
        else:
            return self._data[indices[:n]], self._label[indices[:n]]

    @property
    def data(self):
        return self._data

    @property
    def label(self):
        return self._label
