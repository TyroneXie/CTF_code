# -*- coding: utf-8 -*-
import logging
import os
import pickle
import sys
import time
import warnings
from argparse import ArgumentParser
import numpy as np
import tensorflow as tf
from tfsnippet.examples.utils import MLResults, print_with_title
from tfsnippet.scaffold import VariableSaver
from tfsnippet.utils import get_variables_as_dict, register_config_arguments, Config
from omni_anomaly.training_transfer import Trainer
from omni_anomaly.eval_methods import pot_eval, bf_search
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
    get_file_way = 'test_flow'  # 'train_flow', 'test_flow'
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
    train_start = 0
    max_train_size = None  # `None` means full train set
    batch_size = 250
    l2_reg = 0.0001
    initial_lr = 0.001
    lr_anneal_factor = 0.5
    lr_anneal_epoch_freq = 40
    lr_anneal_step_freq = None
    std_epsilon = 1e-4

    # evaluation parameters
    test_n_z = 1
    test_batch_size = 50
    test_start = 0
    max_test_size = None  # `None` means full test set

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
    level = 0.01

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
    dataset_list = (config.dataset.replace(" ", '')).split(',')
    index_list = [int(i) for i in (config.index.replace(" ", '')).split(',')] if config.index is not None else None
    config.x_dim = get_data_dim(dataset_list)

    # construct the model under `variable_scope` named 'model'
    with tf.variable_scope(config.save_dir) as model_vs:
        model = OmniAnomaly(config=config, name=config.save_dir)

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
                          untrainable_variables_keyvalues=["rnn_p_x","rnn_q_z"], 
                          )

        # construct the predictor
        predictor = Predictor(model, batch_size=config.batch_size, n_z=config.test_n_z,
                              last_point_only=True)

        with tf.Session().as_default():
            # Restore variables from `save_dir`.
            saver = VariableSaver(get_variables_as_dict(model_vs), config.save_dir)
            saver.restore()

            # get score of train set for POT algorithm
            if config.get_score_for_each_machine_flag:
                if config.get_file_way == 'test_flow':
                    (x_train_list, train_timestamp_list, _), (x_test_list, test_timestamp_list, y_test_list), _ = \
                        get_data(dataset_list, start_time=config.get_data_start_time, last_time=config.get_data_last_time,
                                     method=config.get_file_way, number_list=index_list,
                                     result_dir=config.result_dir)
                    for i in range(99, len(test_timestamp_list)):
                        test_data = np.hstack(x_test_list[i-99:i+1]).reshape(-1, 49)
                        test_score, test_z, pred_speed = predictor.get_score(test_data, mode='cluster_test')
                        test_label = np.zeros([test_score.shape[0]])
                        if config.test_score_filename is not None:
                            with open(os.path.join(
                                    config.result_dir, f'{test_timestamp_list[i]}-{config.test_score_filename}'
                            ), 'wb') as file:
                                pickle.dump([dataset_list, test_score, test_label], file)


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

