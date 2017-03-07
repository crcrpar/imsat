from __future__ import print_function
import os
import yaml
import json
import six

import numpy as np
import chainer
from chainer import functions as F
from chainer import links as L
from chainer import training
from chainer.training import extensions

from models.hash_mnist import Encoder
from models.hash_mnist import HashWrapper
from dataset import IMSATIterator
from updater import IMSAT_HASHUpdater


def main():
    encoder = HashWrapper()
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(encoder)
    with open(os.path.join('config', 'mnist_hash.yml'), 'r') as f:
        conf = yaml.load(f)
    if not os.path.isdir(conf['result']):
        os.mkdir(conf['result'])
    print('# SETTING #')
    print(json.dumps(conf, indent=2))
    batch_size = conf['batch_size']
    n_bit = conf['n_bit']
    N_query = conf['num_query']
    if conf['seed']:
        np.random.seed(int(conf['seed']))
    if conf['gpu'] >= 0:
        encoder.to_gpu()
    train, test = chainer.datasets.get_mnist(scale=2.0)
    x_train, x_test = [i[0] for i in train], [i[0] for i in test]
    y_train, y_test = [i[1] for i in train], [i[1] for i in test]
    x = np.concatenate((np.asarray(x_train), np.asarray(x_test))).astype(np.float32)
    print('x.shape: {}'.format(x.shape))
    x = list(x)
    y = np.concatenate((np.asarray(y_train), np.asarray(y_test))).astype(np.int32)
    dataset = chainer.datasets.TupleDataset((x, y))
    iterator = IMSATIterator(x, batch_size=batch_size)
    # iterator = IMSATIterator(dataset, batch_size=batch_size)
    updater = IMSAT_HASHUpdater(iterator, optimizer)
    trainer = training.Trainer(updater, out=conf['result'])
    trainer.extend(extensions.LogReport(keys=['conditional_entropy', 'marginal_entropy', 'pairwise_mi', 'loss_info']))
    trainer.extend(extensions.PrintReport(['epoch', 'iteration', 'main/loss_info']), trigger=(50, 'iteration'))
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.run()


if __name__ == '__main__':
    main()
