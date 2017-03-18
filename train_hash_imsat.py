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
from dataset import PreprocessDataset
from updater import IMSAT_HASHUpdater
from iterators.multi_iterator import MultiDatasetIterator
from iterators.paralleliterator import ParallelMultiDatasetIterator


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
    N_query = conf['num_query']
    if conf['seed']:
        np.random.seed(int(conf['seed']))
    if conf['gpu'] >= 0:
        encoder.to_gpu()

    if conf['root']:
        files = os.listdir(conf['root'])
    else:
        msg = 'set dataset dir'
        raise Exception(msg)
    dataset = PreprocessDataset(files, conf['root'], crop_size=227)
    augmented_dataset = PreprocessDataset(files, conf['root'], random=True)
    iterator = MultiDatasetIterator(dataset, augmented_dataset, batch_size=batch_size)
    updater = IMSAT_HASHUpdater(iterator, (conf['epoch'], 'epoch'), optimizer)
    trainer = training.Trainer(updater, out=conf['result'])
    trainer.extend(extensions.LogReport(keys=['conditional_entropy', 'marginal_entropy', 'pairwise_mi', 'loss_info']))
    trainer.extend(extensions.PrintReport(['epoch', 'iteration', 'main/loss_info']), trigger=(50, 'iteration'))
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.run()


if __name__ == '__main__':
    main()
