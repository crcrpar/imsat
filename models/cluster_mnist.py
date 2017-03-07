#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
import numpy as np
from sklearn import metrics
import chainer
from chainer import functions as F
from chainer import links as L
from munkres import Munkres


class Encoder(chainer.Chain):

    def __init__(self, in_size=784, h0=1200, h1=1200, out_size=10, prop_eps=0.25):
        henormal = chainer.initializers.HeNormal(scale=0.1)
        self._layers = {
            'l1': L.Linear(in_size=in_size, out_size=h0, initialW=henormal),
            'bn1': L.BatchNormalization(size=h0),
            'l2': L.Linear(in_size=h0, out_size=h1, initialW=henormal),
            'bn2': L.BatchNormalization(size=h1),
            'l3': L.Linear(in_size=h1, out_size=out_size)
        }
        self.prop_eps = prop_eps
        self.n_class = out_size
        super(Encoder, self).__init__(**self._layers)

    def forward(self, x, _train=True, _update=True):
        h1 = F.relu(self.bn1(self.l1(x), test=not _train, finetune=_update))
        h2 = F.relu(self.bn2(self.l2(h1), test=not _train, finetune=_update))
        y = self.l3(h2)
        return y


class ClusterWrapper(Encoder):
    """ This class is wrapper of chainer.Chain for IMSAT

    This wrapper for clustering
    """

    def __init__(self):
        super(ClusterWrapper, self).__init__()

    def aux(self, x):
        return self.forward(x, _train=True, _update=False)

    def entropy(self, p):
        if not isinstance(p, chainer.Variable):
            p = chainer.Variable(p)
        if p.data.ndim == 1:
            return - F.sum(p * F.log(p + 1e-8))
        elif p.data.ndim == 2:
            return - F.sum(p * F.log(p + 1e-8)) / float(len(p.data))
        else:
            raise NotImplementedError

    def kl(self, p, q):
        if not isinstance(p, chainer.Variable):
            p = chainer.Variable(p)
            q = chainer.Variable(q)
        kl_d = F.sum(p * F.log((p + 1e-8) / (q + 1e-8))) / float(len(p.data))
        return kl_d

    def distance(self, y0, y1):
        s_y0 = F.softmax(y0)
        s_y1 = F.softmax(y1)
        distance = self.kl(s_y0, s_y1)
        return distance

    def vat(self, x, eps_list, xi=10, Ip=1, _update=True):
        if not isinstance(x, chainer.Variable):
            x = chainer.Variable(x)
        y = self.forward(x, _update=_update)
        y.unchain_backward()
        d = np.random.normal(size=x.data.shape).astype(np.float32)
        d /= np.sqrt(np.sum(d ** 2, axis=1)).reshape((x.shape[0], 1))
        var_d = chainer.Variable(d)
        for ip in range(Ip):
            y2 = self.forward(x + xi * var_d, _update=_update)
            kl_loss = self.distance(y, y2)
            kl_loss.backward()
            d = var_d.grad
            d /= np.sqrt(np.sum(d ** 2, axis=1)).reshape((x.data.shape[0], 1))
        var_d = chainer.Variable(d.astype(np.float32))
        eps = self.prop_eps * eps_list
        y2 = self.forward(x + F.transpose(eps * F.transpose(var_d)))
        return self.distance(y, y2)

    def loss_equal(self, x):
        if not isinstance(x, chainer.Variable):
            x = chainer.Variable(x.astype(np.float32))
        p_logit = self.forward(x)
        p = F.softmax(p_logit)
        p_ave = F.sum(p, axis=0) / x.data.shape[0]
        entropy = self.entropy(p)
        return entropy, - F.sum(p_ave * F.log(p_ave + 1e-8))

    def loss_unlabeled(self, x, eps_list):
        L = self.vat(x, eps_list, _update=False)
        return L

    def loss_test(self, x, t):
        prob = F.softmax(self.forward(x, _train=False)).data
        p_margin = self.xp.sum(prob, axis=0) / len(prob)
        entropy = self.xp.sum(-p_margin * self.xp.log(p_margin + 1e-8))
        prediction = np.argmax(prob, axis=1)
        if isinstance(t, chainer.Variable):
            tt = t.data
        else:
            tt = t
        m = Munkres()
        mat = np.zeros(shape=(self.n_class, self.n_class))
        for i in range(self.n_class):
            for j in range(self.n_class):
                mat[i, j] = np.sum(np.logical_and(prediction == i, tt == j))
        indices = m.compute(-mat)
        corresp = []
        for i in range(self.n_class):
            corresp.append(indices[i][1])
        pred_corresp = [corresp[int(predicted)] for predicted in prediction]
        acc = self.xp.sum(pred_corresp == tt) / float(len(tt))
        return acc, entropy


def main():
    print('test of building ClusterWrapper')
    clustering_encoder = ClusterWrapper()
    print('clustering_encoder')
    print(clustering_encoder.links)


if __name__ == '__main__':
    main()
