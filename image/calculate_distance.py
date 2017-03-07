#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
import os

import numpy as np
import chainer


def calculate_distance(data, _num_data, dst, gpu=-1):
    """ Calculate distance

        Args:
            data: to process
            _num_data: how many nearest items to save
    """
    if gpu >= 0:
        chainer.cuda.get_device(gpu).use()
    xp = chainer.cuda.cupy if gpu >= 0 else np
    if not isinstance(_num_data, list):
        num_data = [_num_data]
    dist_list = [[] for i in range(len(num_data))]

    for i in range(len(data)):
        if i % 1000 == 0:
            print('processing {}th data'.format(i))
        dist = xp.sqrt(xp.sum((data - data[i]) ** 2, axis=1))
        dist[i] = 1000
        sorted_dist = np.sort(dist)
        for j in range(len(num_data)):
            dist_list[j].append(sorted_dist[num_data[j]])
    for i in range(len(num_data)):
        np.savetxt(dst.format(num_data[i]))


def main():
    train, test = chainer.datasets.get_mnist(scale=2.0)
    x_train = np.asarray([i[0] for i in train], dtype=np.float32)
    x_test = np.asarray([i[0] for i in test], dtype=np.float32)
    x = np.concatenate((x_train, x_test), axis=0)
    n_data = 10
    dst = os.path.join('mnist', '{}th_neighbor.txt')
    calculate_distance(x, n_data, dst)


if __name__ == '__main__':
    main()
