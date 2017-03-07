import copy
import six

import chainer
from chainer.dataset import convert


class IMSAT_HASHUpdater(chainer.training.StandardUpdater):

    def __init__(self,
                 iterator,
                 optimizer,
                 converter=convert.concat_examples,
                 device=None,
                 loss_func=None):
        super(IMSAT_HASHUpdater, self).__init__(iterator,
                                                optimizer,
                                                converter=converter,
                                                device=device,
                                                loss_func=loss_func)

    def update_core(self):
        batch, index = self._iterators['main'].next()
        in_arrays = self.converter(batch, self.device)
        print(type(in_arrays))
        x = chainer.Variable(in_arrays)
        optimizer = self._optimizers['main']
        model = optimizer.target
        if not hasattr(model, 'loss_information'):
            msg = 'model should have `loss_information` method'
            raise NotImplementedError(msg)
        # convert batch to chainer.Variable
        cond_ent, marg_ent, pairwise_mi = model.loss_information(in_arrays)
        loss_ul = model.loss_unlabeled(x, index)
        loss_info = cond_ent - marg_ent + pairwise_mi
        model.cleargrads()
        (loss_ul + model.lam * loss_info).backward()
        optimizer.update()
        chainer.report({
                'conditional_entropy': cond_ent,
                'marginal_entropy': marg_ent,
                'pairwise_mi': pairwise_mi,
                'loss_info': loss_info,
                'vat': loss_ul
                }, model)
