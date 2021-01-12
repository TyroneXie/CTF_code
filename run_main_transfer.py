import subprocess
from multiprocessing import Pool
import os
from omni_anomaly.z_cluster import compute_z_matrix, cluster_on_z_matrix
from omni_anomaly.data_getter import datetime_to_timestamp, get_historical_max_min_mean, get_threshold
import pickle
from argparse import ArgumentParser
import sys
import numpy as np
import time
from tfsnippet.utils import register_config_arguments, Config


class ExpConfig_transfer(Config):
    file_prefix = '../result'
    train_start_time = '2020-04-18 00:00:00'
    train_last_time = '2020-04-23 00:00:00'
    train_z_start_time = '2020-04-21 00:00:00'
    train_z_last_time = '2020-04-22 00:00:00'
    test_start_time = '2020-04-23 00:00:10'
    test_last_time = '2020-04-30 23:59:59'
    test_start_timestamp = None
    test_last_timestamp = None

    executable_action = '1,2,3,4,5'  # or "1,2" or  "3",
    action1_model_dir = 'model_0'
    action1_result_dir = f'{file_prefix}/result_for_period1'
    action1_sample_ratio = 0.1
    action1_machine_sample_ratio = 0.166
    action1_GPU_device_number = '-1'

    action2_run_parallel_number = 8

    action3_z_file_dir = 'z_results'
    action3_z_distance_matrix_name = 'z_distance.pkl'
    action3_machine_file_name = 'machine_list.pkl'
    action3_cluster_number = 5
    action3_cluster_png_filename = 'HAC.png'
    action3_cluster_result_filename = 'z_cluster.pkl'

    action4_run_parallel_number = 5
    action4_cluster_max_machine = 30
    action4_model_dir_prefix = 'model'
    action4_result_dir_prefix = f'{file_prefix}/result_for_period2'
    action4_save_path = f'{file_prefix}/'

    action5_run_parallel_number = 5  ############ 10
    action5_get_historical_data_info_flag = True  ########### True
    action5_get_threshold_flag = False  ############## True


def train(dataset, save_dir, result_dir, get_file_way='pkl', get_data_sample_ratio=1.0, get_data_start_time=0,
          get_data_last_time=0, save_z_flag=0, get_score_flag=1, restore_dir=None, max_epoch=10,
          GPU_device_number='-1', untrainable_variables_keyvalues=None, sample_ratio=None, sample_z_ratio=1.0,
          average_flag=False, valid_step_freq=360, index=None, save_model_flag=True, train_flag=True,
          online_flag=False, get_score_on_dim=False):
    if online_flag:
        args = [sys.executable, 'main_transfer_online.py', '--dataset', dataset, '--save_dir', save_dir, '--result_dir', result_dir,
                '--get_data_sample_ratio', str(get_data_sample_ratio), '--get_data_start_time', str(get_data_start_time),
                '--get_data_last_time', str(get_data_last_time), '--get_file_way', get_file_way]
    else:
        args = [sys.executable, 'main_transfer.py', '--dataset', dataset, '--save_dir', save_dir, '--result_dir', result_dir,
                '--get_data_sample_ratio', str(get_data_sample_ratio), '--get_data_start_time', str(get_data_start_time),
                '--get_data_last_time', str(get_data_last_time), '--get_file_way', get_file_way]
    if save_z_flag == 1:
        args.extend(['--save_z', 'True'])
    if get_score_flag == 0:
        args.extend(['--get_score_for_each_machine_flag', 'False'])
    if untrainable_variables_keyvalues is not None:
        args.extend(['--untrainable_variables_keyvalues', untrainable_variables_keyvalues])
    if GPU_device_number != '-1':
        args.extend(['--GPU_device_number', GPU_device_number])
    if restore_dir is not None:
        args.extend(['--restore_dir', restore_dir])
    if max_epoch != 10:
        args.extend(['--max_epoch', str(max_epoch)])
    if sample_ratio is not None:
        args.extend(['--sample_ratio', str(sample_ratio)])
    if average_flag != False:
        args.extend(['--average_flag', 'True'])
    if sample_z_ratio < 1.0:
        args.extend(['--sample_z_ratio', str(sample_z_ratio)])
    if valid_step_freq != 360:
        args.extend(['--valid_step_freq', str(valid_step_freq)])
    if index is not None:
        args.extend(['--index', index])
    if save_model_flag != True:
        args.extend(['--save_model_flag', 'False'])
    if train_flag != True:
        args.extend(['--train_flag', 'False'])
    if get_score_on_dim != False:
        args.extend(['--get_score_on_dim', 'True'])
    # print(f'running program {args}.')
    subprocess.call(args)


def run_train(args):
    train(*args)


def train_part1(dataset, start_time=0, last_time=0, action1_model_dir='model_0',
                action1_result_dir='result_for_period1', action1_GPU_device_number='-1', action1_sample_ratio=None,
                action1_machine_sample_ratio=None):
    if action1_machine_sample_ratio is not None:
        N = int(1.0 / action1_machine_sample_ratio)
        dataset = dataset[::N]
        dataset = ','.join(dataset)
    train(dataset, get_file_way='train_flow', save_dir=action1_model_dir, result_dir=action1_result_dir,
          get_score_flag=0, GPU_device_number=action1_GPU_device_number, get_data_sample_ratio=action1_sample_ratio,
          get_data_start_time=start_time, get_data_last_time=last_time)


def get_all_machines_z(dataset, start_time=0, last_time=0, action1_model_dir='model_0',
                       action1_result_dir='result_for_period1', action2_run_parallel_number=1):
    save_z_flag, get_score_flag, max_epoch = 1, 1, 0
    machine_list = dataset
    new_machine_list = []
    for i in range(int(np.ceil(len(machine_list) / 20.0))):
        new_machine_list.append(','.join(machine_list[i*20:(i+1)*20]))
    args_list = [
        (m, action1_model_dir, action1_result_dir, 'train_flow', 1.0, start_time, last_time,
         save_z_flag, get_score_flag, action1_model_dir, max_epoch, "-1", None, None, 0.1, False, 360,
         None, False)
        for GPU_number, m in enumerate(new_machine_list)
    ]
    pool = Pool(action2_run_parallel_number)
    pool.map(run_train, args_list)


def train_part2(dataset, action1_result_dir='result_for_period1', action3_z_file_dir='z_results',
                action3_z_distance_matrix_name='z_distance.pkl', action3_cluster_number=3,
                action3_cluster_png_filename='HAC.png', action3_cluster_result_filename='z_cluster.pkl',
                action3_machine_file_name='machine_list.pkl'):
    machine_list = dataset
    compute_z_matrix(machine_list, result_dir=action1_result_dir, save_z_dir=action3_z_file_dir,
                     distance_matrix_name=action3_z_distance_matrix_name, machine_file_name=action3_machine_file_name)
    cluster_on_z_matrix(save_z_dir=action3_z_file_dir, distance_matrix_name=action3_z_distance_matrix_name,
                        N=action3_cluster_number, png_filename=action3_cluster_png_filename,
                        cluster_filename=action3_cluster_result_filename)


def train_part3(dataset, start_time=0, last_time=0, action3_z_file_dir='z_results',
                action3_cluster_result_filename='z_cluster.pkl',
                action3_machine_file_name='machine_list.pkl', action4_run_parallel_number=1,
                action4_model_dir_prefix='model',
                action4_result_dir_prefix='result_for_period2', action1_model_dir='model_0',
                action4_cluster_max_machine=50):
    with open(os.path.join(action3_z_file_dir, action3_cluster_result_filename), 'rb') as f:
        cluster_result = pickle.load(f)
    with open(os.path.join(action3_z_file_dir, action3_machine_file_name), 'rb') as f:
        all_machine_list = pickle.load(f)
    machine_list = [[all_machine_list[int(j)] for j in list(i)] for i in cluster_result]

    new_machine_list = []
    for each_machine_list in machine_list:
        if len(each_machine_list) > action4_cluster_max_machine:
            skip_number = int(len(each_machine_list) / action4_cluster_max_machine) + 1
            new_machine_list.append(each_machine_list[::skip_number])
        else:
            new_machine_list.append(each_machine_list)

    print(new_machine_list)
    args_list = [
        (','.join(m), f'{action4_model_dir_prefix}_{i + 1}', f'{action4_result_dir_prefix}_{i + 1}', 'train_flow', 0.1,
         start_time, last_time, 0, 1, action1_model_dir, 10, "-1", "rnn_p_x,rnn_q_z", None, 1.0, False, 100)
        for i, m in enumerate(new_machine_list)
    ]
    pool = Pool(action4_run_parallel_number)
    pool.map(run_train, args_list)


def train_part4(dataset, historical_start_time=0, historical_last_time=0, start_time=0, last_time=0,
                action3_z_file_dir='z_results', action3_cluster_result_filename='z_cluster.pkl',
                action3_machine_file_name='machine_list.pkl', action5_run_parallel_number=1,
                action4_model_dir_prefix='model', action4_save_path='/data00/Omni_transfer_v2/',
                action4_result_dir_prefix='result_for_period2', get_historical_data_info_flag=False,
                get_threshold_flag=False):
    with open(os.path.join(action3_z_file_dir, action3_cluster_result_filename), 'rb') as f:
        cluster_result = pickle.load(f)
    with open(os.path.join(action3_z_file_dir, action3_machine_file_name), 'rb') as f:
        all_machine_list = pickle.load(f)

    if get_historical_data_info_flag:
        get_historical_max_min_mean(historical_start_time, historical_last_time, all_machine_list, action4_save_path)
    machine_list = [[all_machine_list[int(j)] for j in list(i)] for i in cluster_result]
    cluster_list = [','.join([str(int(j)) for j in list(i)]) for i in cluster_result]
    if get_threshold_flag:
        for i in range(len(machine_list)):
            get_threshold(f'{action4_result_dir_prefix}_{i + 1}', level=0.001)

    args_list = [
        (','.join(m_list), f'{action4_model_dir_prefix}_{i + 1}', f'{action4_result_dir_prefix}_{i + 1}', 'test_flow',
         1.0, start_time, last_time, 0, 1, f'{action4_model_dir_prefix}_{i + 1}', 0, "-1", None, None, 1.0, False, 360,
         c_list, False, True, True, True) for i, (m_list, c_list) in enumerate(zip(machine_list, cluster_list))
    ]
    if action5_run_parallel_number == 1:
        pool = Pool(1)
        pool.map(run_train, args_list)
    else:
        pool = Pool(action5_run_parallel_number)
        for arg in args_list:
            pool.apply_async(train, args=arg)
        pool.close()
        pool.join()


def main():
    config = ExpConfig_transfer()

    # parse the arguments
    arg_parser = ArgumentParser()
    register_config_arguments(config, arg_parser)
    arg_parser.parse_args(sys.argv[1:])
    executable_action_list = (config.executable_action.replace(" ", "")).split(',')
    dataset = [str(i) for i in range(533)]

    if '1' in executable_action_list:
        train_part1(dataset, start_time=datetime_to_timestamp(config.train_start_time),
                    last_time=datetime_to_timestamp(config.train_last_time),
                    action1_model_dir=config.action1_model_dir, action1_result_dir=config.action1_result_dir,
                    action1_GPU_device_number=config.action1_GPU_device_number,
                    action1_sample_ratio=config.action1_sample_ratio,
                    action1_machine_sample_ratio=config.action1_machine_sample_ratio)
    if '2' in executable_action_list:
        get_all_machines_z(dataset, start_time=datetime_to_timestamp(config.train_z_start_time),
                           last_time=datetime_to_timestamp(config.train_z_last_time),
                           action1_model_dir=config.action1_model_dir, action1_result_dir=config.action1_result_dir,
                           action2_run_parallel_number=config.action2_run_parallel_number)
    if '3' in executable_action_list:
        train_part2(dataset, action1_result_dir=config.action1_result_dir, action3_z_file_dir=config.action3_z_file_dir,
                    action3_z_distance_matrix_name=config.action3_z_distance_matrix_name,
                    action3_cluster_number=config.action3_cluster_number,
                    action3_cluster_png_filename=config.action3_cluster_png_filename,
                    action3_cluster_result_filename=config.action3_cluster_result_filename,
                    action3_machine_file_name=config.action3_machine_file_name)
    if '4' in executable_action_list:
        train_part3(dataset, start_time=datetime_to_timestamp(config.train_start_time),
                    last_time=datetime_to_timestamp(config.train_last_time),
                    action3_z_file_dir=config.action3_z_file_dir,
                    action3_cluster_result_filename=config.action3_cluster_result_filename,
                    action3_machine_file_name=config.action3_machine_file_name,
                    action4_run_parallel_number=config.action4_run_parallel_number,
                    action4_model_dir_prefix=config.action4_model_dir_prefix,
                    action4_result_dir_prefix=config.action4_result_dir_prefix,
                    action1_model_dir=config.action1_model_dir,
                    action4_cluster_max_machine=config.action4_cluster_max_machine)
    if '5' in executable_action_list:
        if config.test_start_timestamp is None:
            _test_start_timestamp = datetime_to_timestamp(config.test_start_time)
            _test_last_timestamp = datetime_to_timestamp(config.test_last_time)
        else:
            _test_start_timestamp = config.test_start_timestamp
            _test_last_timestamp = config.test_last_timestamp

        train_part4(dataset, historical_start_time=datetime_to_timestamp(config.train_z_start_time),
                    historical_last_time=datetime_to_timestamp(config.train_z_last_time),
                    start_time=_test_start_timestamp,
                    last_time=_test_last_timestamp,
                    action3_z_file_dir=config.action3_z_file_dir,
                    action3_cluster_result_filename=config.action3_cluster_result_filename,
                    action3_machine_file_name=config.action3_machine_file_name,
                    action5_run_parallel_number=config.action5_run_parallel_number,
                    action4_model_dir_prefix=config.action4_model_dir_prefix,
                    action4_save_path=config.action4_save_path,
                    action4_result_dir_prefix=config.action4_result_dir_prefix,
                    get_historical_data_info_flag=config.action5_get_historical_data_info_flag,
                    get_threshold_flag=config.action5_get_threshold_flag)


main()
