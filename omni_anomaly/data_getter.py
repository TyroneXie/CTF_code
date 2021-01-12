import numpy as np
import json
import requests
import os
import pandas as pd
import time
from datetime import datetime
from pandas.io.json import json_normalize
from multiprocessing import Pool
from omni_anomaly.eval_methods import pot_eval_online
import pickle


def new_getdata_from_clickhouse(begin, end, ips, get_one_point_flag=False):
    start_datetime = datetime.fromtimestamp(begin)
    start_index = start_datetime.hour * 120 + start_datetime.minute * 2 + int(np.ceil(start_datetime.second / 30))
    start_date = start_datetime.day
    if start_index == 2880:
        start_index, start_date = 0, start_date + 1
    last_datetime = datetime.fromtimestamp(end)
    last_index = last_datetime.hour * 120 + last_datetime.minute * 2 + int(np.ceil(last_datetime.second / 30))
    last_date = last_datetime.day
    if last_index == 0:
        last_index, last_date = 2880, last_date - 1
    start_date, last_date = max(18, start_date),  min(last_date, 30)

    if len(ips) == 1:
        data_list = []
        for i in range(start_date, last_date + 1):
            try:
                data_list.append(np.loadtxt(f'../CTF_data/{ips[0]}_{i}.txt', delimiter=','))
            except Exception as e:
                print(e)
                continue
        data = np.vstack(data_list)
        data = data[start_index:int(last_index + (last_date - start_date) * 2880), :]
        return data, None, None
    else:
        machine_data_list = []
        for ip in ips:
            data_list = []
            for i in range(start_date, last_date + 1):
                try:
                    data_list.append(np.loadtxt(f'../CTF_data/{ip}_{i}.txt', delimiter=','))
                except Exception as e:
                    print(e)
                    data = np.zeros([2880, 49])
                    data[:, :] = np.nan
                    data_list.append(data)
            data = np.vstack(data_list)
            machine_data_list.append(data[start_index:int(last_index + (last_date - start_date) * 2880), :])
        return machine_data_list, None, None


def get_historical_max_min_mean(begin, end, ips, save_path='/data00/Omni_transfer_v2/', KPI_number=49):
    pool = Pool(20)
    args_list = [(begin, end, [ip]) for ip in ips]
    result_list = pool.map(run_new_getdata_from_clickhouse2, args_list)
    mean_matrix = np.zeros([len(ips), KPI_number])
    std_matrix = np.zeros([len(ips), KPI_number])
    for i, (mean_array, std_array) in enumerate(result_list):
        if mean_array is not None:
            mean_matrix[i, :] = mean_array
            std_matrix[i, :] = std_array
        else:
            mean_matrix[i, :] = np.nan
            std_matrix[i, :] = np.nan
    mean_matrix = np.where(np.isnan(mean_matrix), np.nanmean(mean_matrix, axis=0, keepdims=True), mean_matrix)
    std_matrix = np.where(np.isnan(std_matrix), np.nanmean(std_matrix, axis=0, keepdims=True), std_matrix)
    os.makedirs(os.path.join(save_path, 'historical_data'), exist_ok=True)
    with open(os.path.join(save_path, 'historical_data/data_info.pkl'), 'wb') as f:
        pickle.dump([mean_matrix, std_matrix], f)


def run_new_getdata_from_clickhouse2(args):
    try:
        KPI_matrix, index_list, kpi = new_getdata_from_clickhouse(*args)
        print(f'{args} finished.')
        mean_array = np.nanmean(KPI_matrix, axis=0, keepdims=True)
        std_array = np.nanstd(KPI_matrix, axis=0, keepdims=True)
        KPI_matrix = np.where(KPI_matrix > mean_array + 20 *std_array, mean_array + 20 *std_array, KPI_matrix)
        KPI_matrix = np.where(KPI_matrix < mean_array - 20 *std_array, mean_array - 20 *std_array, KPI_matrix)
        return np.nanmean(KPI_matrix, axis=0), np.nanstd(KPI_matrix, axis=0)
    except:
        print(args)
        return None, None


def run_getdata_from_clickhouse(start_time, last_time, dataset):
    try:
        return new_getdata_from_clickhouse(start_time, last_time, dataset, True)
    except:
        print(start_time, last_time, dataset)
        return None, None, None


def run_new_getdata_from_clickhouse(args):
    try:
        return new_getdata_from_clickhouse(*args)
    except Exception as e:
        print(e)
        return None, None, None


def datetime_to_timestamp(_datetime):
    assert len(_datetime) == 19
    return int(datetime.strptime(_datetime, '%Y-%m-%d %H:%M:%S').timestamp())


def get_threshold(filepath, level=0.001, q=1e-4, save_score=True):
    score_list, valid_score_list = [], []
    for f in os.listdir(filepath):
        if 'train_score' in f:
            with open(os.path.join(filepath, f), 'rb') as fr:
                data = pickle.load(fr)
                valid_number = int(data.shape[0] * 0.3)
                score_list.append(data[-valid_number:])
                valid_score_list.append(data[:-valid_number])
    train_score = np.hstack(score_list)
    valid_score = np.hstack(valid_score_list)
    threshold = pot_eval_online(train_score, valid_score, q=q, level=level)
    threshold = min(-10000, threshold)

    print(filepath)
    print(f'training set anomaly ratio: {1.0 * np.sum(train_score < threshold) / train_score.shape[0]}')
    print(f'validation set anomaly ratio: {1.0 * np.sum(valid_score < threshold) / valid_score.shape[0]}')
    if save_score:
        with open(os.path.join(filepath, 'train_threshold.pkl'), 'wb') as fw:
            pickle.dump(threshold, fw)
    return threshold, 1.0 * np.sum(train_score < threshold) / train_score.shape[0], \
           1.0 * np.sum(valid_score < threshold) / valid_score.shape[0]

