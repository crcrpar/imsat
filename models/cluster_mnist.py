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

    def __init__(self, in_size=784, h0=1200, h1=1200, out_size=10, prop_eps=1.0):
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

    def unit_vector(self, v):
        v /= (self.xp.sqrt(self.xp.sum(v ** 2, axis=1)).reshape((-1, 1)) + 1e-16)
        return v

    def vat(self, x, xi=10, Ip=1):
        if not isinstance(x, chainer.Variable):
            x = chainer.Variable(x)
        y1 = self.forward(x, _train=False)
        y1.unchain_backward()

        d = self.xp.random.normal(size=x.shape, dtype=np.float32)
        d = d / self.xp.sqrt(self.xp.sum(d ** 2, axis=1)).reshape((x.shape[0], 1))
        for ip in range(Ip):
            d_var = chainer.Variable(d.astype(np.float32))
            y2 = self.forward(x + xi * d_var)
            kl_d = self.kl(y1, y2)
            kl_d.backward()
            d = d_var.grad
            d = self.unit_vector(d)
        d_var = chainer.Variable(d.astype(np.float32))
        y2 = self.forward(x + self.prop_eps * d_var, _train=False)
        vat_value = self.distance(y1, y2)
        return vat_value

    def loss_equal(self, x):
        if not isinstance(x, chainer.Variable):
            x = chainer.Variable(x.astype(np.float32))
        p_logit = self.forward(x)
        p = F.softmax(p_logit)
        p_ave = F.sum(p, axis=0) / x.data.shape[0]
        entropy = self.entropy(p)
        return entropy, - F.sum(p_ave * F.log(p_ave + 1e-8))

    def loss_unlabeled(self, x):
        L = self.vat(x)
        return L

    def loss_test(self, x, t):
        xp = self.xp
        prob = F.softmax(self.forward(x, _train=False)).data
        p_margin = xp.sum(prob, axis=0) / len(prob)
        entropy = xp.sum(-p_margin * xp.log(p_margin + 1e-8))
        prediction = np.argmax(prob, axis=1)
        if isinstance(t, chainer.Variable):
            tt = t.data
        else:
            tt = t
        m = Munkres()
        mat = np.zeros(shape=(self.n_class, self.n_class))
        for i in range(self.n_class):
            for j in range(self.n_class):
                mat[i, j] = xp.sum(np.logical_and(prediction == i, tt == j))
        indices = m.compute(-mat)
        corresp = []
        for i in range(self.n_class):
            corresp.append(indices[i][1])
        pred_corresp = [corresp[int(predicted)] for predicted in prediction]
        acc = xp.sum(pred_corresp == tt) / float(len(tt))
        chainer.report({'acc': chainer.Variable(acc), 'entropy': chainer.Variabl(entropy)}, self)
        return acc, entropy


def main():
    print('test of building ClusterWrapper')
    clustering_encoder = ClusterWrapper()
    print('clustering_encoder')
    print(clustering_encoder.links)


if __name__ == '__main__':
    main()
