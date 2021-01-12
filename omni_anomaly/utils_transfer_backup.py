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
        result_dir = result_dir.split('/')[-1]
        result_list = run_getdata_from_clickhouse(start_time, last_time, dataset)
        if not os.path.exists(os.path.join('/data00/Omni_transfer/historical_data', result_dir, f'normalize_matrix.pkl')):
            with open('/data00/Omni_transfer/historical_data/data_info.pkl', 'rb') as f:
                [mean_matrix, max_matrix, min_matrix] = pickle.load(f)
                min_matrix = min_matrix[number_list, :]
                max_matrix = max_matrix[number_list, :]
                mean_matrix = mean_matrix[number_list, :]
        else:
            with open(os.path.join('/data00/Omni_transfer/historical_data', result_dir, f'normalize_matrix.pkl'), 'rb') as f:
                [mean_matrix, max_matrix, min_matrix] = pickle.load(f)
        matrix = np.zeros([len(dataset), 49])
        now_time = 0
        for i, (KPI, _time, KPI_list) in enumerate(result_list):
            if KPI is None:
                matrix[i, :] = np.nan
            else:
                matrix[i, :] = KPI
                now_time = _time[0]
        matrix = np.where(np.isnan(matrix), mean_matrix, matrix)
        max_matrix = np.where(matrix > max_matrix, matrix, max_matrix)
        min_matrix = np.where(matrix < min_matrix, matrix, min_matrix)
        mean_matrix = (1999 * mean_matrix + matrix) / 2000
        mean_matrix = np.where(mean_matrix > max_matrix, max_matrix, mean_matrix)
        mean_matrix = np.where(mean_matrix < min_matrix, min_matrix, mean_matrix)
        matrix = (matrix - min_matrix) / (max_matrix - min_matrix + 1e-9)
        matrix = np.nan_to_num(matrix)
        matrix[matrix > 1.0] = 1.0
        matrix[matrix < 0.0] = 0.0
        os.makedirs(os.path.join('/data00/Omni_transfer/historical_data', result_dir), exist_ok=True)
        with open(os.path.join('/data00/Omni_transfer/historical_data', result_dir, f'{now_time}.pkl'), 'wb') as fw:
            pickle.dump(matrix, fw)
        with open(os.path.join('/data00/Omni_transfer/historical_data', result_dir, f'normalize_matrix.pkl'), 'wb') as f:
            pickle.dump([mean_matrix, max_matrix, min_matrix], f)
        filedir = sorted(os.listdir(os.path.join('/data00/Omni_transfer/historical_data', result_dir)))
        filedir = [f for f in filedir if f[:-4] < str(now_time)]
        if len(filedir) < 100:
            raise ValueError('No historical data.')
        else:
            historical_filedir = filedir[-99:]
            matrix_list = []
            for f in historical_filedir:
                with open(os.path.join('/data00/Omni_transfer/historical_data', result_dir, f), 'rb') as fr:
                    last_matrix = pickle.load(fr)
                    matrix_list.append(last_matrix)
            matrix_list.append(matrix)
            matrix_all = np.hstack(matrix_list)
            matrix_all = matrix_all.reshape(-1, 49)
            test_data_list = [matrix_all]
            test_timestamp_list = [now_time]

    return (train_data_list, train_timestamp_list, None), (test_data_list, test_timestamp_list, test_label_list), KPI_list


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
    df = MinMaxScaler().fit_transform(df)
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