import os
import random

import numpy
try:
    from PIL import Image
    available = True
except ImportError as e:
    available = False
    _import_error = e

import chainer
from chainer.datasets import ImageDataset


def _read_image_as_array(size, path, dtype):
    f = Image.open(path)
    try:
        image = numpy.asarray(f, dtype=dtype)
    finally:
        if hasattr(f, 'close'):
            f.close()
    return image


class Dataset(ImageDataset):

    def __init__(self, paths, root='.', dtype=numpy.float32):
        self.size = 227
        super(Dataset, self).__init__(paths, root, dtype)

    def get_example(self, i):
        path = os.path.join(self._root, self._paths[i])
        image = _read_image_as_array(self.size, path, self._dtype)

        if image.ndim == 2:
            image = image[:, :, numpy.newaxis]
        return image.transpose(2, 0, 1)


class PreprocessDataset(chainer.dataset.DatasetMixin):

    def __init__(self, path, root, crop_size, mean=None, random=False):
        self.base = chainer.datasets.ImageDataset(path, root)
        if mean is not None:
            self.mean = mean.astype('f')
        else:
            self.mean = None
        self.crop_size = crop_size
        self.random = random

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        crop_size = self.crop_size

        image = self.base[i]
        _, h, w = image.shape

        if self.random:
            top = random.randint(0, h - crop_size - 1)
            left = random.randint(0, w - crop_size - 1)
            if random.randint(0, 1):
                image = image[:, :, ::-1]
        else:
            top = (h - crop_size) // 2
            left = (w - crop_size) // 2
        bottom = top + crop_size
        right = left + crop_size
        image = image[:, top:bottom, left:right]
        if self.mean is not None:
            image -= self.mean[:, top:bottom, left:right]
        return image
