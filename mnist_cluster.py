# !/usr/bin/env python
# coding: utf-8

from __future__ import print_function
import os
import datetime
import json
import yaml

import numpy as np
import chainer
from chainer import functions as F
from chainer import links as L
from munkres import Munkres

from models.cluster_mnist import Encoder
from models.cluster_mnist import ClusterWrapper
from load_mnist import DataGenerator


def main():
    encoder = ClusterWrapper()
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(encoder)
    with open(os.path.join('config', 'mnist_clustering.yml'), 'r') as f:
        conf = yaml.load(f)
    print(json.dumps(conf, indent=2))
    if not os.path.isdir(conf['result']):
        os.mkdir(conf['result'])
    if conf['gpu'] >= 0:
        encoder.to_gpu()
    train, test = chainer.datasets.get_mnist(scale=2.0)
    x_train, y_train = [i[0] for i in train], [i[1] for i in train]
    x_test, y_test = [i[0] for i in test], [i[1] for i in test]
    data = np.concatenate((np.asarray(x_train), np.asarray(x_test)))
    data -= np.ones_like(data)
    label = np.concatenate((np.asarray(y_train), np.asarray(y_test)))
    n_data = data.shape[0]
    iter_range = int(n_data / conf['batch_size'])
    dataset = DataGenerator(data=data, label=label)
    if not os.path.exists(os.path.join(conf['dataset'], conf['nearest'].format(conf['K']))):
        from calculate_distance import calculate_distance
        _dst = os.path.join(conf['dataset'], conf['nearest'].format(conf['K']))
        calculate_distance(dataset, conf['num_data'], _dst, conf['gpu'])
    nearest_dist = np.loadtxt(os.path.join(conf['dataset'], conf['nearest'].format(conf['K']))).astype(np.float32)
    log_report = {}
    log_report['timestamp'] = datetime.datetime.now().strftime('%Y/%m/%d')
    log_report['log'] = []
    for epoch in range(conf['epoch']):
        tmp_report = {}
        tmp_report['epoch'] = epoch
        print('# epoch: {}'.format(epoch))
        sum_loss_entropy_max, sum_loss_entropy_min, vatt = .0, .0, .0
        for _iter in range(iter_range):
            x_u, _, ind = dataset.get(conf['batch_size'], need_index=True)
            loss_equal1, loss_equal2 = encoder.loss_equal(x_u)
            loss_equal = loss_equal1 - conf['mu'] * loss_equal2
            sum_loss_entropy_min += loss_equal1.data
            sum_loss_entropy_max += loss_equal2.data
            loss_unlabeled = encoder.loss_unlabeled(x_u, nearest_dist[ind])
            optimizer.target.cleargrads()
            (loss_unlabeled + conf['lam'] * loss_equal).backward()
            optimizer.update()
            vatt += loss_unlabeled.data
            loss_unlabeled.unchain_backward()
        tmp_report['max_entropy'] = sum_loss_entropy_max / iter_range
        tmp_report['min_entropy'] = sum_loss_entropy_min / iter_range
        tmp_report['vatt'] = vatt / iter_range
        x_ul, t_ul = dataset.data, dataset.label
        x_test = chainer.Variable(x_ul, volatile=True)
        t_test = chainer.Variable(t_ul, volatile=True)
        acc, ment = encoder.loss_test(x_test, t_test)
        tmp_report['test'] = {'ment': ment, 'accuracy': acc}
        log_report['log'].append(tmp_report)
        print(json.dumps(tmp_report, ensure_ascii=False, indent=2))
    time_stamp = datetime.datetime.now().strftime('%Y/%m/%d_%H%M')
    with open(os.path.join(conf['result'], '{}.json'.format(time_stamp)), 'w') as f:
        json.dump(log_report, f)
    model_path = os.path.join(conf['result'], time_stamp + '.npz')
    if encoder.xp == chainer.cuda.cupy:
        encoder.to_cpu()
    chainer.serializers.save_npz(model_path, encoder)


if __name__ == '__main__':
    main()
