import os
import sys
import yaml

import numpy as np
import chainer
from dataset import PreprocessDataset


def compute_mean(dataset):
    sum_image = 0.0
    N = len(dataset)
    for i, image in enumerate(dataset):
        sum_image += image
        sys.stderr.write('{} / {}\r'.format(i, N))
        sys.stderr.flush()
    sys.stederr.write('\n')
    return sum_image / N


def main():
    with open('compute_mean.yml', 'r') as f:
        conf = yaml.load(f)
    root = conf['root']
    files = pd.read_csv(conf['item_ids'])
    files = files.as_matrix(columns=['item_id'])
    files = list(files)
    dataset = PreprocessDataset(files, root, 227)
    mean = compute_mean(dataset)
    np.save(conf['out'], mean)


if __name__ == '__main__':
    main()
