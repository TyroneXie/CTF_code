# -*- coding: utf-8 -*-
import os
import pickle
from omni_anomaly.data_getter import new_getdata_from_clickhouse, run_getdata_from_clickhouse, run_new_getdata_from_clickhouse
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from multiprocessing import Pool
import time

prefix = "processed"


def save_z(z, filename='z'):
    """
    save the sampled z in a txt file

    """
    with open(filename + '.pkl', 'wb') as file:
        pickle.dump(z[:, -1, :3], file)


def get_data_dim(dataset):
    if dataset == 'SMAP':
        return 25
    elif dataset == 'MSL':
        return 55
    elif dataset[0].startswith('machine'):
        return 38
    else:
        # raise ValueError('unknown dataset '+str(dataset))
        return 49


def get_data(dataset, do_preprocess=True, start_time=None, last_time=None, sample_ratio=1.0, method='pkl',
             average_flag=False, number_list=None, result_dir=None):
    """
    get data from pkl files

    return shape: (([train_size, x_dim], [train_size] or None), ([test_size, x_dim], [test_size]))
    """
    x_dim = get_data_dim(dataset)

    np.random.seed(2020)
    train_data_list, train_timestamp_list, test_data_list, test_timestamp_list, test_label_list = [], [], [], [], []
    if method == 'pkl':
        for number, ds in enumerate(dataset):
            with open(os.path.join(prefix, ds + '_train.pkl'), "rb") as f:
                train_data = pickle.load(f).reshape((-1, x_dim))
                if do_preprocess:
                    train_data_list.append(preprocess(train_data))
                else:
                    train_data_list.append(train_data)
                print(f"dataset {number} train set shape: ", train_data.shape)
            with open(os.path.join(prefix, ds + '_test.pkl'), "rb") as f:
                test_data = pickle.load(f).reshape((-1, x_dim))
                if do_preprocess:
                    test_data_list.append(preprocess(test_data))
                else:
                    test_data_list.append(test_data)
                print(f"dataset {number} test set shape: ", test_data.shape)
            if os.path.exists(os.path.join(prefix, ds + '_test_label.pkl')):
                with open(os.path.join(prefix, ds + '_test_label.pkl'), "rb") as f:
                    test_label_data = pickle.load(f).reshape((-1))
                    test_label_list.append(test_label_data)
            else:
                test_label_list.append(None)
        train_timestamp_list = None
        test_timestamp_list = None
        KPI_list = None

    elif method == 'train_flow':
        time_length = last_time - start_time
        random_time_length = int((1-sample_ratio) * time_length)
        sample_time_length = int(time_length - random_time_length)
        args_list = []
        for number, ds in enumerate(dataset):
            np.random.seed(number)
            new_start_time = int(start_time + random_time_length * np.random.random())
            new_last_time = new_start_time + sample_time_length
            args_list.append((new_start_time, new_last_time, [ds]))
        if len(args_list) >= 20:
            pool = Pool(5)
            result_list = pool.map(run_new_getdata_from_clickhouse, args_list)
        else:
            result_list = []
            for args in args_list:
                try:
                    result = new_getdata_from_clickhouse(*args)
                    result_list.append(result)
                except:
                    result_list.append((None, None, None))
        for ds, (_data, _timestamp, KPI_list) in zip(dataset, result_list):
            if (_data is None) or (_data.shape[1] != 49):
                print(f'wrong machine: {ds}.')
                continue
            print(f'machine {ds} shape: {_data.shape}')
            _data = preprocess(_data)
            if average_flag:
                average_number = int(_data.shape[0] / 5)
                _data = _data[:int(average_number * 5), :]
                _timestamp = _timestamp[:average_number]
                _data = _data.reshape(average_number, 5, _data.shape[1])
                _data = np.mean(_data, axis=1)
            train_data_list.append(_data)
            train_timestamp_list.append(_timestamp)
            test_data_list.append(None)
            test_timestamp_list.append(None)
            test_label_list.append(None)
    else:
        save_path = '/'.join(result_dir.split('/')[:-1])
        data_start_time = start_time - 30 * 99
        data_list, _, _ = run_getdata_from_clickhouse(data_start_time, last_time, dataset)
        with open(os.path.join(save_path, 'historical_data/data_info.pkl'), 'rb') as f:
            [mean_matrix, std_matrix] = pickle.load(f)
            mean_matrix = mean_matrix[number_list, :]
            std_matrix = std_matrix[number_list, :]

        time_length = data_list[0].shape[0]
        data_all = np.vstack(data_list)
        now_time = data_start_time
        for i in range(time_length):
            matrix = data_all[i::time_length, :]
            matrix = np.where(np.isnan(matrix), mean_matrix, matrix)
            matrix = np.where(matrix > mean_matrix + 20 *std_matrix, mean_matrix + 20 *std_matrix, matrix)
            matrix = np.where(matrix < mean_matrix - 20 *std_matrix, mean_matrix - 20 *std_matrix, matrix)
            old_mean_matrix = mean_matrix
            mean_matrix = (1999 * mean_matrix + matrix) / 2000
            std_matrix = np.sqrt((1999 * (old_mean_matrix ** 2 + std_matrix ** 2) + matrix ** 2) / 2000 - mean_matrix ** 2)

            matrix = (matrix - mean_matrix) / (std_matrix + 1e-9)
            matrix = np.nan_to_num(matrix)
            matrix[matrix > 20.0] = 20.0
            matrix[matrix < -20.0] = -20.0
            test_timestamp_list.append(now_time)
            test_data_list.append(matrix)
            now_time += 30

    return (train_data_list, train_timestamp_list, None), (test_data_list, test_timestamp_list, test_label_list), None


def preprocess(df):
    """returns normalized and standardized data.
    """

    df = np.asarray(df, dtype=np.float32)

    if len(df.shape) == 1:
        raise ValueError('Data must be a 2-D array')

    if np.any(sum(np.isnan(df)) != 0):
        print('Data contains null values. Will be replaced with 0')
        df = np.nan_to_num(df)

    # normalize data
    # df = MinMaxScaler().fit_transform(df)
    mean_array = np.mean(df, axis=0, keepdims=True)
    std_array = np.std(df, axis=0, keepdims=True)
    df = np.where(df > mean_array + 20 *std_array, mean_array + 20 *std_array, df)
    df = np.where(df < mean_array - 20 *std_array, mean_array - 20 *std_array, df)
    df = (df - np.mean(df, axis=0, keepdims=True)) / (np.std(df, axis=0, keepdims=True) + 1e-9)
    if np.any(sum(np.isnan(df)) != 0):
        df = np.nan_to_num(df)
    print('Data normalized')

    return df


def minibatch_slices_iterator(length, batch_size,
                              ignore_incomplete_batch=False):
    """
    Iterate through all the mini-batch slices.

    Args:
        length (int): Total length of data in an epoch.
        batch_size (int): Size of each mini-batch.
        ignore_incomplete_batch (bool): If :obj:`True`, discard the final
            batch if it contains less than `batch_size` number of items.
            (default :obj:`False`)

    Yields
        slice: Slices of each mini-batch.  The last mini-batch may contain
               less indices than `batch_size`.
    """
    start = 0
    stop1 = (length // batch_size) * batch_size
    while start < stop1:
        yield slice(start, start + batch_size, 1)
        start += batch_size
    if not ignore_incomplete_batch and start < length:
        yield slice(start, length, 1)


class BatchSlidingWindow(object):
    """
    Class for obtaining mini-batch iterators of sliding windows.

    Each mini-batch will have `batch_size` windows.  If the final batch
    contains less than `batch_size` windows, it will be discarded if
    `ignore_incomplete_batch` is :obj:`True`.

    Args:
        array_size (int): Size of the arrays to be iterated.
        window_size (int): The size of the windows.
        batch_size (int): Size of each mini-batch.
        excludes (np.ndarray): 1-D `bool` array, indicators of whether
            or not to totally exclude a point.  If a point is excluded,
            any window which contains that point is excluded.
            (default :obj:`None`, no point is totally excluded)
        shuffle (bool): If :obj:`True`, the windows will be iterated in
            shuffled order. (default :obj:`False`)
        ignore_incomplete_batch (bool): If :obj:`True`, discard the final
            batch if it contains less than `batch_size` number of windows.
            (default :obj:`False`)
    """

    def __init__(self, array_size, window_size, batch_size, excludes=None,
                 shuffle=False, ignore_incomplete_batch=False):
        # check the parameters
        if window_size < 1:
            raise ValueError('`window_size` must be at least 1')
        if array_size < window_size:
            raise ValueError('`array_size` must be at least as large as '
                             '`window_size`')
        if excludes is not None:
            excludes = np.asarray(excludes, dtype=np.bool)
            expected_shape = (array_size,)
            if excludes.shape != expected_shape:
                raise ValueError('The shape of `excludes` is expected to be '
                                 '{}, but got {}'.
                                 format(expected_shape, excludes.shape))

        # compute which points are not excluded
        if excludes is not None:
            mask = np.logical_not(excludes)
        else:
            mask = np.ones([array_size], dtype=np.bool)
        mask[: window_size - 1] = False
        where_excludes = np.where(excludes)[0]
        for k in range(1, window_size):
            also_excludes = where_excludes + k
            also_excludes = also_excludes[also_excludes < array_size]
            mask[also_excludes] = False

        # generate the indices of window endings
        indices = np.arange(array_size)[mask]
        self._indices = indices.reshape([-1, 1])

        # the offset array to generate the windows
        self._offsets = np.arange(-window_size + 1, 1)

        # memorize arguments
        self._array_size = array_size
        self._window_size = window_size
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._ignore_incomplete_batch = ignore_incomplete_batch

    def get_iterator(self, arrays):
        """
        Iterate through the sliding windows of each array in `arrays`.

        This method is not re-entrant, i.e., calling :meth:`get_iterator`
        would invalidate any previous obtained iterator.

        Args:
            arrays (Iterable[np.ndarray]): 1-D arrays to be iterated.

        Yields:
            tuple[np.ndarray]: The windows of arrays of each mini-batch.
        """
        # check the parameters
        arrays = tuple(np.asarray(a) for a in arrays)
        if not arrays:
            raise ValueError('`arrays` must not be empty')

        # shuffle if required
        if self._shuffle:
            np.random.shuffle(self._indices)

        # iterate through the mini-batches
        for s in minibatch_slices_iterator(
                length=len(self._indices),
                batch_size=self._batch_size,
                ignore_incomplete_batch=self._ignore_incomplete_batch):
            idx = self._indices[s] + self._offsets
            yield tuple(a[idx] if len(a.shape) == 1 else a[idx, :] for a in arrays)