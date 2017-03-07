from sklearn.datasets import fetch_mldata
import numpy as np


class Data:

    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.index = np.arange(len(data))

    def get_index_data(self, index_list):
        return self.data[index_list]

    def get(self, n, need_index=False):
        ind = np.random.permutation(self.data.shape[0])
        if need_index:
            return self.data[ind[:n], :].astype(np.float32), self.label[ind[:n]].astype(np.int32), self.index[ind[:n]].astype(np.int32)
        else:
            return self.data[ind[:n], :].astype(np.float32), self.label[ind[:n]].astype(np.int32)


def load_mnist_whole(scale, shift, PATH='.'):
    print 'fetch MNIST dataset'
    mnist = fetch_mldata('MNIST original', data_home=PATH)
    mnist.data = mnist.data.astype(np.float32) * scale + shift
    mnist.target = mnist.target.astype(np.int32)
    whole = Data(mnist.data, mnist.target)

    print "load mnist done", whole.data.shape
    return whole
