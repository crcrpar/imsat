# wrapper for encoder
This is a wrapper for you to tray IMSAT for your own encoders.
The encoders must have `forward` function and args `_train (bool)` and `_update (bool)`.
You should set options for `L.BatchNormalization`, i.e., `test (bool)` and `finetune (bool)`.
Sample model is in `models.py` and please follow its `forward` function.

## Usage
Define your `Encoder (chainer.Chain)` in a file and copy `HashWrapper` and/or `ClusterWrapper` from `ChainWrapper` to the file.

#### e.g.
```python
#!/usr/bin/env python
# coding: utf-8
import numpy as np
from sklearn import metrics
import chainer
from chainer import functions as F
from chainer import links as L

class Encoder(chainer.Chain):

    def __init__(self, in_size=784, h0=1200, h1=1200, out_size=10, prop_eps=0.25):
        henormal = chainer.initializers.HeNormal(scale=0.1)
        self._layers = {
            ...
        }
        self.prop_eps = prop_eps
        self.n_class = out_size
        super(Encoder, self).__init__(**self._layers)

    def forward(self, x, _train=True, _update=True):
        ...
        y = self.l3(h2)
        return y
        
        
class FooWrapper(Encoder):

    def __init__(self):
        super(FooWrapper, self).__init__()
    ...
```
