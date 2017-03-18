#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
import os
import copy
import yaml
import json
import six

import numpy as np
import chainer
from chainer import training
from chainer.training import extensions
from models.cluster_mnist import Encoder
from models.cluster_mnist import ClusterWrapper
from updater import IMSAT_CLUSTERUpdater


class ClusterEvaluator(extensions.Evaluator):

    def evaluate(self):
        ret = super(ClusterEvaluator, self).evavluate()
        return ret


def main():
    with open(os.path.join('config', 'mnist_clustering.yml'), 'r') as f:
        conf = yaml.load(f)
    print('# settings')
    print(json.dumps(conf, indent=2))
    encoder = ClusterWrapper(conf['K'], conf['eps'])
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(encoder)
    if conf['gpu']:
        chainer.cuda.get_device(0).use()
        encoder.to_gpu()
    dataset, _ = chainer.datasets.get_mnist(withlabel=False)
    iterator = chainer.iterators.SerialIterator(dataset)
    _, test = chainer.datasets.get_mnist()
    test_iter = chainer.iterators.SerialIterator(test)
    updater = IMSAT_CLUSTERUpdater(iterator, optimizer, device=conf['gpu'])
    print_interval = (25), 'iteration'
    snapshot_interval = (1), 'epoch'
    trainer = training.Trainer(updater, (conf['epoch'], 'epoch'), out=conf['result'])
    trainer.extend(extensions.LogReport(
        keys=['max_entropy', 'min_entropy', 'vat']))
    trainer.extend(extensions.observe_lr(), trigger=print_interval)
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'main/max_entropy', 'main/min_entropy', 'max/vat', 'lr'], trigger=print_interval))
    trainer.extend(extensions.snapshot(), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        encoder, 'model_epoch_{.updater.epoch}'), trigger=snapshot_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(ClusterEvaluator(test_iter, encoder,
                                    conf['gpu']), trigger=snapshot_interval)
    if conf['resume']:
        chainer.serializers.load_npz(conf['resume'], trainer)
    trainer.run()


if __name__ == '__main__':
    main()
