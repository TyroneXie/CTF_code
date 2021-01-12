import os
import pickle
import numpy as np
import pandas as pd
from omni_anomaly.eval_methods import pot_eval, calc_seq


label_result_all = '../label_result'


def evaluate_results(machine_data_filename, file_number, init_score_length=720, q=1e-3, level=0.02):
    df = pd.DataFrame(columns=['ip', 'best_f', 'precision', 'recall', 'TP', 'TN', 'FP', 'FN', 'threshold'])
    with open(f'{machine_data_filename}/{file_number}_score.pkl', 'rb') as fr:
        score_matrix, _, ip_list2 = pickle.load(fr)
    for ip in ip_list2:
        _score = score_matrix[:, ip_list2.index(ip)]
        with open(f'{label_result_all}/{ip}.pkl', 'rb') as fr:
            _label, _ = pickle.load(fr)
        init_score = _score[:init_score_length]
        p_t, pot_th, _, _ = pot_eval(
            init_score,
            _score[init_score_length:], _label[init_score_length:],
            q=q, level=level, threshold=-100
        )
        df.loc[df.shape[0], :] = [ip] + list(p_t) + [pot_th]
    df.loc['mean', :] = df.mean()
    return df


def main(filename=1, dirname='transfer', score_dir='label_data'):
    os.makedirs(f'POT_result/{dirname}/{filename}', exist_ok=True)
    for q in [1e-5]:
        for level in [0.03, 0.02, 0.01, 0.015]:
            df = evaluate_results(f'{score_dir}', filename, 2880, q, level)
            print(q, level, df.loc['mean', :])
            df.to_csv(f'POT_result/{dirname}/{filename}/{q}_{level}.csv')


def read_results(filename='../POT_result/transfer/1'):
    result_df = pd.DataFrame(columns=['p', 'level', 'f-score', 'ave-precision', 'ave-recall', 'number'])
    for file in os.listdir(filename):
        p, level = file.split('_')[0], (file.split('_')[1])[:-4]
        df = pd.read_csv(os.path.join(filename, file))
        df = df.iloc[:-1, :]
        all_machine_number = df.shape[0]
        average_precision, average_recall = np.mean(df['precision']), np.mean(df['recall'])
        result_df.loc[result_df.shape[0], :] = [p, level, 2.0 / (1 / average_precision + 1 / average_recall),
                                                average_precision, average_recall, df.shape[0]]
    print(result_df, '\n', all_machine_number)


def read_results_from_different_cluster(name, threshold_list=None):
    if threshold_list is None:
        param_list = [(1, 1e-5, 0.01), (2, 1e-5, 0.01), (3, 1e-5, 0.03), (4, 1e-5, 0.02), (5, 1e-5, 0.02)]
    else:
        param_list = [(i+1, 1e-5, j) for i, j in enumerate(threshold_list)]
    df_list = []
    all_list = []
    for cluster, param1, param2 in param_list:
        df = pd.read_csv(f'POT_result/{name}/{cluster}/{param1}_{param2}.csv', index_col=0)
        df = df.iloc[:-1, :]
        df['cluster'] = cluster
        all_list += [str(int(i)) for i in df.ip.tolist()]
        df_list.append(df)
    print(sorted(all_list))
    df_all = pd.concat(df_list).reset_index(drop=True)
    df_all = df_all.sort_values(by=['ip']).reset_index(drop=True)
    df_all.loc['mean', :] = df_all.mean()
    df_all = np.round(df_all, 3)
    df_all = df_all.rename(columns={'best_f': 'f1_score'})
    print(df_all.to_string(), df_all.loc['mean', :], 2.0 / (1/df_all.loc['mean', 'precision'] + 1/df_all.loc['mean', 'recall']))
    TP_all, FP_all, FN_all = df_all.loc['mean', 'TP'], df_all.loc['mean', 'FP'], df_all.loc['mean', 'FN']
    precision_all, recall_all = TP_all / (TP_all + FP_all), TP_all / (TP_all + FN_all)
    print(precision_all, recall_all, 2 * precision_all * recall_all / (precision_all + recall_all))


def get_all_score(filename='../label_data_model0', selected_ip_list=None):
    score_matrix_list, ip_list = [], []
    for file_number in os.listdir(filename):
        if not file_number.startswith('.'):
            with open(f'{filename}/{file_number}', 'rb') as fr:
                score_matrix, _, ip_list2 = pickle.load(fr)
                score_matrix_list.append(score_matrix)
                ip_list += ip_list2
    selected_ip_list = list(set(ip_list).intersection(set(selected_ip_list)))
    score_matrix = np.hstack(score_matrix_list).T
    index_list = [ip_list.index(ip) for ip in selected_ip_list]
    return score_matrix[index_list, :], selected_ip_list


(_filename, _data_dir, param_list) = ('CTF', 'label_data', [0.01, 0.01, 0.01, 0.01, 0.015])
for ii in range(1, 6):
    main(ii, _filename, _data_dir)
    read_results(f'POT_result/{_filename}/{ii}')
read_results_from_different_cluster(_filename, param_list)
