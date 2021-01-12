# -*- coding: utf-8 -*-
import logging
import os
import pickle
import sys
import time
import warnings
from argparse import ArgumentParser
from pprint import pformat, pprint
import numpy as np
import tensorflow as tf
from tfsnippet.examples.utils import MLResults, print_with_title
from tfsnippet.scaffold import VariableSaver
from tfsnippet.utils import get_variables_as_dict, register_config_arguments, Config

from omni_anomaly.training_transfer import Trainer
from omni_anomaly.model_transfer import OmniAnomaly
from omni_anomaly.prediction import Predictor
from omni_anomaly.utils_transfer import get_data_dim, get_data, save_z
import sys

import warnings
warnings.filterwarnings("ignore")


class ExpConfig(Config):
    GPU_device_number = "-1"  # CUDA_VISIBLE_DEVICES
    # dataset configuration, optional: one machine or many machines.
    dataset = "machine-1-1,machine-1-2"  # or "machine-1-1"
    index = None
    get_file_way = 'pkl'  # 'train_flow', 'test_flow'
    get_data_start_time = 0
    get_data_last_time = 0
    get_data_sample_ratio = 1.0
    sample_z_ratio = 1.0
    average_flag = False

    # model architecture configuration
    use_connected_z_q = True
    use_connected_z_p = True

    # model parameters
    z_dim = 3
    rnn_cell = 'GRU'  # 'GRU', 'LSTM' or 'Basic'
    rnn_num_hidden = 500
    window_length = 100
    dense_dim = 500
    posterior_flow_type = 'nf'  # 'nf' or None
    nf_layers = 20  # for nf
    max_epoch = 10
    batch_size = 250
    l2_reg = 0.0001
    initial_lr = 0.001
    lr_anneal_factor = 0.5
    lr_anneal_epoch_freq = 40
    lr_anneal_step_freq = None
    std_epsilon = 1e-4

    # evaluation parameters
    test_n_z = 100
    test_batch_size = 50

    # the range and step-size for score for searching best-f1
    # may vary for different dataset
    bf_search_min = -400.
    bf_search_max = 400.
    bf_search_step_size = 1.

    valid_step_freq = 360
    gradient_clip_norm = 10.

    early_stop = True  # whether to apply early stop method

    # pot parameters
    # recommend values for `level`:
    # SMD group 1: 0.0050
    # SMD group 2: 0.0075
    # SMD group 3: 0.0001
    level = 0.002

    # outputs config
    save_z = False  # whether to save sampled z in hidden space
    get_score_on_dim = False  # whether to get score on dim. If `True`, the score will be a 2-dim ndarray
    save_dir = 'model'
    save_model_flag = True
    restore_dir = None  # If not None, restore variables from this dir
    result_dir = 'result_step1'  # Where to save the result file
    train_score_filename = 'train_score.pkl'
    test_score_filename = 'test_score.pkl'
    get_score_for_each_machine_flag = True
    untrainable_variables_keyvalues = None  # 'rnn_q_z','vae','posterior_flow','rnn_p_x','p_x_given_z/x'


def main():
    if config.GPU_device_number != "-1":
        os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU_device_number
    logging.basicConfig(
        level='INFO',
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    new_untrainable_variables_keyvalues = (config.untrainable_variables_keyvalues.replace(" ", '')).split(',') \
        if config.untrainable_variables_keyvalues is not None else None
    dataset_list = (config.dataset.replace(" ", '')).split(',')
    index_list = [int(i) for i in (config.index.replace(" ", '')).split(',')] if config.index is not None else None
    config.x_dim = get_data_dim(dataset_list)

    # prepare the data
    if config.get_file_way == 'pkl':
        (x_train_list, train_timestamp_list, _), (x_test_list, test_timestamp_list, y_test_list), KPI_list = \
            get_data(dataset_list, method=config.get_file_way)
    if 'flow' in config.get_file_way:
        (x_train_list, train_timestamp_list, _), (x_test_list, test_timestamp_list, y_test_list), KPI_list = \
            get_data(dataset_list, start_time=config.get_data_start_time, last_time=config.get_data_last_time,
                     sample_ratio=config.get_data_sample_ratio, method=config.get_file_way,
                     average_flag=config.average_flag, number_list=index_list, result_dir=config.result_dir)

    # construct the model under `variable_scope` named 'model'
    with tf.variable_scope(config.restore_dir) if config.restore_dir is not None \
            else tf.variable_scope(config.save_dir) as model_vs:
        model = OmniAnomaly(config=config, name=config.save_dir) if config.restore_dir is None \
            else OmniAnomaly(config=config, name=config.restore_dir)
        # construct the trainer
        trainer = Trainer(model=model,
                          model_vs=model_vs,
                          max_epoch=config.max_epoch,
                          batch_size=config.batch_size,
                          valid_batch_size=config.test_batch_size,
                          initial_lr=config.initial_lr,
                          lr_anneal_epochs=config.lr_anneal_epoch_freq,
                          lr_anneal_factor=config.lr_anneal_factor,
                          grad_clip_norm=config.gradient_clip_norm,
                          valid_step_freq=config.valid_step_freq,
                          untrainable_variables_keyvalues=new_untrainable_variables_keyvalues
                          )

        # construct the predictor
        predictor = Predictor(model, batch_size=config.batch_size, n_z=config.test_n_z,
                              last_point_only=True)

        with tf.Session().as_default():
            if config.restore_dir is not None:
                # Restore variables from `save_dir`.
                saver = VariableSaver(get_variables_as_dict(model_vs), config.restore_dir)
                saver.restore()

            if config.max_epoch > 0:
                # train the model
                train_start = time.time()
                best_valid_metrics = trainer.fit(x_train_list, valid_portion=0.1)
                train_time = (time.time() - train_start) / config.max_epoch
                best_valid_metrics.update({
                    'train_time': train_time
                })
            else:
                best_valid_metrics = {}

            # get score of train set for POT algorithm
            if config.get_score_for_each_machine_flag:
                if config.get_file_way == 'train_flow':
                    st = time.time()
                    for ds, x_train, train_timestamp in zip(dataset_list, x_train_list, train_timestamp_list):
                        train_score, train_z, train_pred_speed = predictor.get_score(x_train, sample_ratio=config.sample_z_ratio)
                        if config.train_score_filename is not None:
                            with open(os.path.join(config.result_dir, f'{ds}-{config.train_score_filename}'), 'wb') as file:
                                pickle.dump(train_score, file)
                            # with open(os.path.join(config.result_dir, f'{ds}-train_timestamp.pkl'), 'wb') as file:
                            #     pickle.dump(train_timestamp[int(config.window_length-1):], file)
                        if config.save_z:
                            save_z(train_z, os.path.join(config.result_dir, f'{ds}-train_z'))
                    print(f'testing {len(dataset_list)} machine entities cost {time.time() - st}')

            if (config.save_dir is not None) & (config.save_model_flag):
                # save the variables
                var_dict = get_variables_as_dict(model_vs)
                if config.restore_dir is not None:
                    var_dict = {k.replace(config.restore_dir, config.save_dir): i for k, i in var_dict.items()}
                saver = VariableSaver(var_dict, config.save_dir)
                saver.save()
            print('=' * 30 + 'result' + '=' * 30)
            pprint(best_valid_metrics)


if __name__ == '__main__':
    # get config obj
    config = ExpConfig()

    # parse the arguments
    arg_parser = ArgumentParser()
    register_config_arguments(config, arg_parser)
    arg_parser.parse_args(sys.argv[1:])

    # open the result object and prepare for result directories if specified
    results = MLResults(config.result_dir)
    results.save_config(config)  # save experiment settings for review
    results.make_dirs(config.save_dir, exist_ok=True)
    with warnings.catch_warnings():
        # suppress DeprecationWarning from NumPy caused by codes in TensorFlow-Probability
        warnings.filterwarnings("ignore", category=DeprecationWarning, module='numpy')
        main()
