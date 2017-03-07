#!/usr/bin/env python
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
from sklearn import metrics
from munkres import Munkres

from load_mnist import DataGenerator
from models.hash_mnist import Encoder
from models.hash_mnist import HashWrapper


def prepare_data(dataset):
    """ Prepare `query` and `gallary`"""

    data, target = dataset.data, dataset.label
    perm = np.random.permutation(len(target))
    cnt_query = [0] * 10
    idx_query, idx_gallary = [], []
    for i in range(len(target)):
        l = target[perm[i]]
        if cnt_query[l] < 100:
            idx_query.append(perm[i])
            cnt_query[l] += 1
        else:
            idx_gallary.append(perm[i])
    x_query, y_query = np.asarray(data[idx_query]).astype(np.float32), np.asarray(target[idx_query]).astype(np.int32)
    x_gallary, y_gallary = np.asarray(data[idx_gallary]).astype(np.float32), np.asarray(target[idx_gallary]).astype(np.int32)
    query = DataGenerator(x_query, y_query)
    gallary = DataGenerator(x_gallary, y_gallary)
    return query, gallary


def main():
    encoder = HashWrapper()
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(encoder)
    # load settings
    with open(os.path.join('config', 'mnist_hash.yml'), 'r') as f:
        conf = yaml.load(f)
    if not os.path.isdir(conf['result']):
        os.mkdir(conf['result'])
    print('# information')
    print(json.dumps(conf, indent=2))
    batch_size = conf['batch_size']
    lam = conf['lam']
    n_bit = conf['n_bit']
    N_query = conf['num_query']
    if conf['seed']:
        np.random.seed(int(conf['seed']))
    if conf['gpu'] >= 0:
        encoder.to_gpu()
    # prepare dataset
    train, test = chainer.datasets.get_mnist(scale=2.0)
    x_train, x_test = [i[0] for i in train], [i[0] for i in test]
    y_train, y_test = [i[1] for i in train], [i[1] for i in test]
    x = np.concatenate((np.asarray(x_train), np.asarray(x_test))).astype(np.float32)
    x -= np.ones_like(x)
    N, dim = x.shape
    y = np.concatenate((np.asarray(y_train), np.asarray(y_test))).astype(np.int32)
    dataset = DataGenerator(x, y)
    query, gallary = prepare_data(dataset)
    x_query, y_query = query.data, query.label
    x_gallary, y_gallary = gallary.data, gallary.label
    if conf['gpu'] >= 0:
        chainer.cuda.get_device(conf['gpu']).use()
    N_gallary = len(gallary.data)
    if not os.path.exists(os.path.join(conf['dataset'], conf['nearest'].format(conf['K']))):
        from calculate_distance import calculate_distance
        dst = os.path.join(conf['dataset'], conf['nearest'])
        calculate_distance(dataset, conf['K'], dst, conf['gpu'])
    nearest_dist = np.loadtxt(os.path.join(
        conf['dataset'], conf['nearest'].format(conf['K']))).astype(np.float32)
    log_report = {}
    log_report['log'] = []
    iter_range = int(N_gallary / batch_size)
    for epoch in range(conf['epoch']):
        tmp_report = {}
        tmp_report['epoch'] = epoch
        sum_cond_ent, sum_marg_ent, sum_pairwise_mi, sum_vat = 0, 0, 0, 0
        for it in range(iter_range):
            x, _, ind = dataset.get(batch_size, need_index=True)
            cond_ent, marg_ent, pairwise_mi = encoder.loss_information(x, n_bit)
            sum_cond_ent += cond_ent.data
            sum_marg_ent += marg_ent.data
            sum_pairwise_mi += pairwise_mi.data
            loss_info = cond_ent - marg_ent + pairwise_mi
            loss_ul = encoder.loss_unlabeled(x, nearest_dist[ind], conf['xi'], conf['Ip'])
            sum_vat += loss_ul.data
            optimizer.target.cleargrads()
            (loss_ul + lam * loss_info).backward()
            optimizer.update()
            loss_ul.unchain_backward()
            loss_info.unchain_backward()
        condent = sum_cond_ent / iter_range
        margent = sum_marg_ent / iter_range
        pairwise = sum_pairwise_mi / iter_range
        tmp_report['conditional_entropy'] = condent
        tmp_report['marginal_entropy'] = margent
        tmp_report['pairwise_mi'] = pairwise
        tmp_report['vat_loss'] = sum_vat / iter_range
        log_report['log'].append(tmp_report)
        x_query_test = chainer.Variable(x_query, volatile=True)
        y_query_test = chainer.Variable(y_query, volatile=True)
        x_gallary_test = chainer.Variable(x_gallary, volatile=True)
        y_gallary_test = chainer.Variable(y_gallary, volatile=True)
        MAP, withNpreclabel, withRpreclabel = encoder.loss_test(x_query_test, y_query_test, x_gallary_test, y_gallary_test, N_query)
        tmp_report['MAP'] = MAP
        tmp_report['withNpreclabel'] = withNpreclabel
        tmp_report['withRpreclabel'] = withRpreclabel
        log_report['log'].append(tmp_report)
        print(json.dump(tmp_report, indent=2))
    time_stamp = datetime.datetime.now().strftime('%Y%m%d')
    with open(os.path.join(conf['result'], 'hash_{}.json'.format(time_stamp)), 'w') as f:
        json.dump(log_report, f)
    model_path = os.path.join(conf['result'], 'hash_{}.npz'.format(time_stamp))
    if encoder.xp == chainer.cuda.cupy:
        encoder.to_cpu()
    chainer.serializers.save_npz(model_path, encoder)
    print('model saved at {}'.format(model_path))


if __name__ == '__main__':
    main()
