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

    def __init__(self, iterator, target, converter=chainer.dataset.concat_examples,
                 device=None, eval_hook=None, eval_func=None):
        if not hasattr(target, 'loss_test'):
            return False
        super(ClusterEvaluator, self).__init__(
            iterator, target, converter, device, eval_hook, eval_func)

    def evaluate(self):
        iterator = self._iterators['main']
        target = self._targets['main']
        eval_func = self.eval_func or target.loss_test

        if self.eval_hook:
            self.eval_hook(self)
        it = copy.copy(iterator)
        summary = chainer.reporter.DictSummary()

        for batch in it:
            observation = {}
            with chainer.reporter.report_scope(observation):
                in_arrays = self.converter(batch, self.device)
            if isinstance(in_arrays, tuple):
                in_vars = tuple(chainer.Variable(x, volatile='on')
                                for x in in_arrays)
                eval_func(*in_vars)
            elif isinstance(in_arrays, dict):
                in_vars = {key: chainer.Variable(x, volatile='on')
                           for key, x in six.iteritems(in_arrays)}
                eval_func(**in_vars)
            else:
                in_var = chainer.Variable(in_arrays, volatile='on')
                eval_func(in_var)

            summary.add(observation)


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
    trainer = training.Trainer(updater, out=conf['result'])
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
