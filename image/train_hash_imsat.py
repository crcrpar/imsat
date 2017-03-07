from __future__ import print_function
import os
import yaml
import json
import six

import numpy as np
import pandas as pd
import chainer
from chainer import functions as F
from chainer import links as L
from chainer import training
from chainer.training import extensions

from models.hash_mnist import Encoder
from models.hash_mnist import HashWrapper
from dataset import PreprocessDataset
from iterator import IMSATIterator
from updater import IMSAT_HASHUpdater


def main():
    with open(os.path.join('config', 'hash.yml'), 'r') as f:
        conf = yaml.load(f)
    if not os.path.isdir(conf['result']):
        os.mkdir(conf['result'])
    print('# SETTING #')
    print(json.dumps(conf, indent=2))
    # set up model
    encoder = HashWrapper(xi=conf['xi'],
                          Ip=conf['Ip'],
                          N_query=conf['n_query'])
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(encoder)
    # set up dataset
    batch_size = conf['batch_size']
    if conf['seed']:
        np.random.seed(int(conf['seed']))
    if conf['gpu'] >= 0:
        encoder.to_gpu()
    root_data = conf['root']
    files = os.listdir(root_data)
    files = pd.read_csv(conf['item_ids'])
    files = list(files)
    # set up [dataset, iterator, updater]
    dataset = PreprocessDataset(files, root_data)
    iterator = IMSATIterator(dataset, batch_size=batch_size)
    updater = IMSAT_HASHUpdater(iterator, optimizer)
    trainer = training.Trainer(updater, out=conf['result'])
    # set extensions interval
    log_interval = (100), 'iteration'
    snapshot_interval = (1), 'epoch'
    # set Trainer extensions
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.PrintReport(['epoch', 'iteration', 'main/loss_info', 'lr']), trigger=(50, 'iteration'))
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(extensions.snapshot(), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        encoder, 'model_iter_{.updater.epoch}'), trigger=snapshot_interval)
    if conf['resume']:
        chainer.serializers.load_npz(conf['resume'], trainer)
    trainer.run()


if __name__ == '__main__':
    main()
