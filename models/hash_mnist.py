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

    def __init__(self, in_size=784, h0=1200, h1=1200, out_size=16, prop_eps=0.25, lam=0.1):
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
        self.n_bit = out_size
        super(Encoder, self).__init__(**self._layers)

    def forward(self, x, _train=True, _update=True):
        h1 = F.relu(self.bn1(self.l1(x), test=not _train, finetune=_update))
        h2 = F.relu(self.bn2(self.l2(h1), test=not _train, finetune=_update))
        y = self.l3(h2)
        return y


class HashWrapper(Encoder):
    """ This class is wrapper of chainer.Chain for IMSAT"""

    def __init__(self, in_size=784, h0=1200, h1=1200, out_size=16, prop_eps=1.0, lam=0.1):
        super(HashWrapper, self).__init__(in_size, h0, h1, out_size, prop_eps, lam)

    def aux(self, x):
        prob = self.forward(x, _train=True, _update=False)
        return prob

    def entropy(self, p):
        if not isinstance(p, chainer.Variable):
            p = chainer.Variable(p)
        if p.data.ndim == 1:
            return - F.sum(p * F.log(p + 1e-8))
        elif p.data.ndim == 2:
            return - F.sum(p * F.log(p + 1e-8)) / float(len(p.data))
        else:
            msg = 'Not appropriate data shape'
            raise NotImplementedError(msg)

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

    def vat(self, x, eps_list, xi, Ip, _update=True):
        xp = self.xp
        y = self.forward(x, _update=_update)
        y.unchain_backward()
        d = xp.random.normal(size=x.data.shape).astype(np.float32)
        d /= xp.sqrt(xp.sum(d ** 2, axis=1)).reshape((x.data.shape[0], 1))
        var_d = chainer.Variable(d)
        for ip in range(Ip):
            y2 = self.forward(x + xi * var_d, _update=_update)
            kl_loss = self.distance(y, y2)
            kl_loss.backward()
            d = var_d.grad
            d /= xp.sqrt(xp.sum(d ** 2, axis=1)).reshape((x.data.shape[0], 1))
        var_d = chainer.Variable(d.astype(np.float32))
        eps = self.prop_eps * eps_list
        y2 = self.forward(x + F.transpose(eps * F.transpose(var_d)))
        distance = self.distance(y, y2)
        return distance

    def loss_unlabeled(self, x, eps_list, xi, Ip):
        _L = self.vat(x, eps_list, xi, Ip, _update=False)
        return _L

    def loss_test(self, x_query, y_query, x_gallary, y_gallary, N_query):
        query_hash = F.sigmoid(self.forward(x_query, _train=False)).data > 0.5
        gallary_hash = F.sigmoid(self.forward(x_gallary, _train=False)).data > 0.5
        y_query = y_query.data
        y_gallary = y_gallary.data
        withinN_precision_label = 0
        withinR_precision_label = 0
        MAP = 0
        for i in range(N_query):
            hamming_distance = self.xp.sum((1 - query_hash[i]) == gallary_hash, axis=1)
            MAP += metrics.average_precision_score(y_gallary == y_query[i], 1.0 / (1.0 + hamming_distance))
            nearestN_index = np.argsort(hamming_distance)[:500]
            withinN_precision_label += float(np.sum(y_gallary[nearestN_index] == y_query[i])) / 500
            withinR_label = y_gallary[hamming_distance < 3]
            num_withinR = len(withinR_label)
            if not num_withinR == 0:
                withinR_precision_label += self.xp.sum(withinR_label == y_query[i]) / float(num_withinR)
        return MAP / N_query, withinN_precision_label / N_query, withinR_precision_label / N_query

    def loss_information(self, x, n_bit=None):
        if n_bit is None:
            n_bit = self.n_bit
        def _make_p_stacked(p_pair, axis):
            if not isinstance(p_pair, chainer.Variable):
                p_pair = chainer.Variable(p_pair)
            p_stacked = F.concat((p_pair, 1 - p_pair, p_pair, 1 - p_pair), axis=axis)
            return p_stacked
        p_logit = self.forward(x)
        p = F.sigmoid(p_logit)
        p_ave = F.sum(p, axis=0) / x.data.shape[0]
        cond_ent = F.sum(-p * F.log(p + 1e-8) - (1 - p) * F.log(1 - p + 1e-8)) / p.data.shape[0]
        marg_ent = F.sum(-p_ave * F.log(p_ave + 1e-8) - (1 - p_ave) * F.log(1 - p_ave + 1e-8))
        p_ave = F.reshape(p_ave, (1, len(p_ave.data)))
        p_ave_separated = F.separate(p_ave, axis=1)
        p_separated = F.separate(F.expand_dims(p, axis=2), axis=1)
        p_ave_list_i, p_ave_list_j = [], []
        p_list_i, p_list_j = [], []
        for i in range(n_bit - 1):
            p_ave_list_i.extend(list(p_ave_separated[i + 1:]))
            p_list_i.extend(list(p_separated[i + 1:]))
            p_ave_list_j.extend([p_ave_separated[i] for n in range(n_bit - i - 1)])
            p_list_j.extend([p_separated[i] for n in range(n_bit - i - 1)])
        p_ave_pair_i = F.expand_dims(F.concat(tuple(p_ave_list_i), axis=0), axis=1)
        p_ave_pair_j = F.expand_dims(F.concat(tuple(p_ave_list_j), axis=0), axis=1)
        p_pair_i = F.expand_dims(F.concat(tuple(p_list_i), axis=1), axis=2)
        p_pair_j = F.expand_dims(F.concat(tuple(p_list_j), axis=1), axis=2)
        p_pair_stacked_i = _make_p_stacked(p_pair_i, axis=2)
        p_pair_stacked_j = _make_p_stacked(p_pair_j, axis=2)
        p_ave_pair_stacked_i = _make_p_stacked(p_ave_pair_i, axis=1)
        p_ave_pair_stacked_j = _make_p_stacked(p_ave_pair_j, axis=1)
        p_product = F.sum(p_pair_stacked_i * p_pair_stacked_j, axis=0) / len(p.data)
        p_ave_product = p_ave_pair_stacked_i * p_ave_pair_stacked_j
        pairwise_mi = 2 * F.sum(p_product * F.log((p_product + 1e-8) / (p_ave_product + 1e-8)))
        return cond_ent, marg_ent, pairwise_mi


def main():
    print('test of building HashWrapper')
    hash_encoder = HashWrapper()
    print('hash_encoder')
    print(hash_encoder.links)


if __name__ == '__main__':
    main()
