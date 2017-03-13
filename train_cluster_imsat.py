#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
import os
import yaml
import json
import six

import numpy as np
import scipy.ndimage.interpolation.rotate as rotate
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from dataset import RotateDataset

def main():
    encoder = ClusterWrapper()
    optimizer = chaienr.optimizers.Adam()
    optimizer.setup(encoder)

