from __future__ import division
import multiprocessing
from multiprocessing import sharedctypes
import threading
import warnings

import numpy
import six

from chainer.dataset import iterator


class ParallelMultiDatasetIterator(iterator.Iterator):

    """Multi Dataset iterator that loads examples in parallel.

    Note that this iterator effectively prefetches the examples for the next
    batch asynchrononously after the current batch is returned.

    Args:
        dataset1, dataset2 (~chainer.dataset.Dataset)
        batch_size (int)
        repeat (bool)
        shuffle (bool)
        n_processes (int)
        n_prefetch (int)
        shared_mem (int)

    """

    _last_signal = object()

    def __init__(self, dataset1, dataset2, batch_size,  repeat=True, shuffle=True,
                 n_processes=None, n_prefetch=1, shared_mem=None):
        assert len(dataset1) == len(dataset2)
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.batch_size = batch_size
        self._repeat = repeat
        if shuffle:
            self._order = numpy.random.permutation(len(dataset1))
        else:
            self._order = None
        self._prefetch_order = None  # used at the end of each epoch

        self.current_position = 0
        self.epoch = 0
        self.is_new_epoch = False
        self._published_position = None  # initialized at the first iteration

        self.n_processes = n_processes or multiprocessing.cpu_count()
        self.n_prefetch = max(n_prefetch, 1)
        self._shared_mem_size = shared_mem

    def __del__(self):
        self.finalize()

    def __next__(self):
        if not self._repeat and self.epoch > 0:
            raise StopIteration

        self.is_new_epoch = False
        if self._finalized is None:
            self._init()  # start workers
            # load for the first iteration
            for _ in six.moves.range(self.n_prefetch):
                self._invoke_prefetch()

        batch1, batch2 = self._get()
        self._invoke_prefetch()  # prefetch for the next iteration
        return batch1, batch2

    next = __next__

    @property
    def epoch_detail(self):
        return self.epoch + self.current_position / len(self.dataset1)

    def finlize(self):
        if self._finalized is None or self._finalized.is_set():
            return

        self._finalized.set()
        self._ordered_data_queue1.put(self._last_signal)
        self._ordered_data_queue2.put(self._last_signal)
        self._data_queue1.put((-1, -1, -1))
        self._data_queue2.put((-1, -1, -1))
        for _ in self._workers:
            self._index_queue.put((-1, -1, -1))  # termination sognal

        for worker in self._workers:
            worker.join()
        self._get_data_loop_thread.join()

    def serialize(self, serializer):
        self.current_position = serializer('current_position',
                                           self.current_position)
        self.epoch = serializer('epoch', self.epoch)
        self.is_new_epoch = serializer('is_new_epoch', self.is_new_epoch)
        serializer('order', self._order)

    def _init(self):
        finalized = threading.Event()
        self._index_queue = multiprocessing.Queue()
        self._data_queue1 = multiprocessing.Queue()
        self._data_queue2 = multiprocessing.Queue()
        self._ordered_data_queue1 = six.moves.queue.Queue()
        self._ordered_data_queue2 = six.moves.queue.Queue()
        self._unused_mem_queue = six.moves.queue.Queue()
        self._mem_list = []
        self._cnt = 0

        self._workers = []

        if self._shared_mem_size is not None:
            self._init_process()

        self._get_data_loop_thread = threading.Thread(
            target=_get_data_loop, name="get_data_loop",
            args=(self._data_queue1, self._data_queue2,
                  self._ordered_data_queue1, self._ordered_data_queue2,
                  self._mem_list,
                  self._unused_mem_queue,
                  finalized, self._last_signal))
        self._get_data_loop_thread.daemon = True
        self._get_data_loop_thread.start()

        self._finalized = finalized

    def _init_process(self):
        assert len(self._workers) == 0
        assert self._shared_mem_size is not None
        mem_size = self._shared_mem_size
        for i in six.moves.range(self.batch_size * (self.n_prefetch + 1)):
            self._mem_list.append(sharedctypes.RawArray('b', mem_size))
            self._unused_mem_queue.put(i)

        args = (self.dataset1, self.dataset2, self._index_queue,
                self._data_queue1, self._data_queue2,
                self._mem_list)
        for _ in range(self.n_processes):
            worker = multiprocessing.Process(target=_worker, args=args)
            worker.daemon = True
            self._workers.append(worker)
            worker.start()

    def _invoke_prefetch(self):
        n = len(self.dataset1)
        i = self._pushed_position
        if i is None:  # first iteration
            i = self.current_position

        order = self._order
        measure_mode = len(self._workers) == 0
        max_size = 0
        for _ in six.moves.range(self.batch_size):
            if i >= n:
                if not self._repeat:
                    break
                i = 0
                if order is not None:
                    # cannot shuffle the order directly here,
                    # since the iterator may be serialized
                    # before the prefetched data are consumed by the user,
                    # in which case an inconsistency appears.
                    order = order.copy()
                    numpy.random.shuffle(order)
            index = i if order is None else order[i]
            if measure_mode:
                data1 = self.dataset1[index]
                data2 = self.dataset2[index]
                max_size = max(max_size, _measure(data1))
                self._data_queue1.put((self._cnt, None, data1))
                self._data_queue2.put((self._cnt, None, data2))
                del data1, data2
            else:
                self._index_queue.put(
                    (self._cnt, self._unused_mem_queue.get(), index))
            self._cnt += 1
            i += 1

        self._prefetch_order = order  # Temporarily store the shuffled order.
        self._pushed_position = i

        if measure_mode:
            self._shared_mem_size = max_size
            self._init_process()

    def _get(self):
        n = len(self.dataset1)
        i = self.current_position

        batch1 = []
        batch2 = []
        for _ in six.moves.range(self.batch_size):
            d1 = self._ordered_data_queue1.get()
            d2 = self._ordered_data_queue2.get()
            if (d1 is self._last_signal) and (d2 is self._last_signal):
                break
            batch1.append(d1)
            batch2.append(d2)
            i += 1
            if i >= n:
                self.epoch += 1
                self.is_new_epoch = True
                if not self._repeat:
                    break
                i = 0
        self.current_position = i
        # Eventually overwrite the (possibly shuffled) order.
        self._order = self._prefetch_order
        return batch1, batch2


def _get_data_loop(data_queue1, data_queue2,
                   ordered_data_queue1, ordered_data_queue2,
                   mem_list, unused_mem_queue,
                   finalized, last_signal):
    buf1 = {}
    buf2 = {}
    cnt = 0
    while not finalized.is_set():
        if (cnt in buf1) and (cnt in buf2):
            data1 = buf1.pop(cnt)
            data2 = buf2.pop(cnt)
        else:
            try:
                c1, mem_index1, data1 = data_queue1.get(timeout=0.5)
                c2, mem_index2, data2 = data_queue2.get(timeout=0.5)
                if mem_index1 == mem_index2:
                    mem_index = mem_index1
                else:
                    msg = 'Inconsistency occured'
                    raise ValueError(msg)
            except six.moves.queue.Empty:
                continue
            if c1 < 0 and c2 < 0:
                break
            if mem_index is not None:
                data1 = _unpack(data1, mem_list[mem_index])
                data2 = _unpack(data2, mem_list[mem_index])
                unused_mem_queue.put(mem_index)
            if c1 != cnt and c2 != cnt:
                buf1[c1] = data1
                buf2[c2] = data2
                continue
        ordered_data_queue1.put(data1)
        ordered_data_queue2.put(data2)
        del data1, data2
        cnt += 1
    ordered_data_queue1.put(last_signal)
    ordered_data_queue2.put(last_signal)


class _PackedNdarray(object):

    def __init__(self, array, mem, offset):
        self.shape = array.shape
        self.dtype = array.dtype
        self.nbytes = array.nbytes
        self.size = array.size
        self.offset = offset
        total = self.offset + self.nbytes
        if total > len(mem):
            raise ValueError(
                'Shared memory size is too small. expect:{}, actual:{}'.format(
                    total, len(mem)))
        target = numpy.frombuffer(mem, self.dtype, self.size, self.offset)
        target[...] = array.ravel()

    def unpack(self, mem):
        ret = numpy.frombuffer(mem, self.dtype, self.size, self.offset)
        ret = ret.reshape(self.shape).copy()
        return ret


def _measure(data):
    expect = 0
    t = type(data)
    if t is tuple or t is list or t is dict:
        for v in data:
            if isinstance(v, numpy.ndarray):
                expect += v.nbytes
    return expect


def _pack(data, mem):
    if len(mem) == 0:
        return data
    t = type(data)
    offset = 0
    over = False
    if t is tuple or t is list:
        ret = []
        for v in data:
            if isinstance(v, numpy.ndarray):
                if v.nbytes + offset > len(mem):
                    over = True
                else:
                    v = _PackedNdarray(v, mem, offset)
                    offset += v.nbytes
            ret.append(v)
        data = t(ret)
    elif t is dict:
        ret = {}
        for k, v in six.iteritems(data):
            if isinstance(v, numpy.ndarray):
                if v.nbytes + offset > len(mem):
                    over = True
                else:
                    v = _PackedNdarray(v, mem, offset)
                    offset += v.nbytes
            ret[k] = v
        data = ret
    if over:
        expect = _measure(data)
        warnings.warn(
            'Shared memory size is too small.\n' +
            'Please set shared_mem option for MultiprocessIterator.\n' +
            'Expect shared memory size: {} bytes.\n'.format(expect) +
            'Actual shared memory size: {} bytes.'.format(len(mem)),
            UserWarning)
    return data


def _unpack(data, mem):
    if len(mem) == 0:
        return data
    t = type(data)
    if t is tuple or t is list:
        ret = []
        for v in data:
            if isinstance(v, _PackedNdarray):
                v = v.unpack(mem)
            ret.append(v)
        data = t(ret)
    elif t is dict:
        ret = {}
        for k, v in six.iteritems(data):
            if isinstance(v, _PackedNdarray):
                v = v.unpack(mem)
            ret[k] = v
        data = ret
    return data


def _worker(dataset, in_queue, out_queue, mem_list):
    while True:
        cnt, mem_index, index = in_queue.get()
        if cnt < 0:
            break
        mem = mem_list[mem_index]
        data = _pack(dataset[index], mem)
        out_queue.put((cnt, mem_index, data))
    out_queue.close()
    out_queue.join_thread()
