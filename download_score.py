
import pickle
import os
import numpy as np

result_filedir = '../result'
all_start_time = 1587571210
all_last_time = 1588262350


def get_score_or_matrix_data(number=2):
    filepath = f'{result_filedir}/result_for_period2_{number}'
    f_list = sorted([f for f in os.listdir(filepath) if 'test_score' in f])
    score_list, timestamp_list = [], []
    for f in f_list:
        try:
            timestamp = int(f.split('-')[0])
            if (timestamp >= all_start_time) and (timestamp <= all_last_time):
                with open(os.path.join(filepath, f), 'rb') as fr:
                    ip_list, score, label = pickle.load(fr)
                    score_list.append(np.nansum(score, axis=1))
                    timestamp_list.append(timestamp)
        except:
            continue
    score_matrix = np.array(score_list)
    return score_matrix, timestamp_list, ip_list


def save_file(number=2):
    os.makedirs(f'label_data', exist_ok=True)
    score_matrix, timestamp_list, ip_list = get_score_or_matrix_data(number)
    with open(f'label_data/{number}_score.pkl', 'wb') as fw:
        pickle.dump([score_matrix, timestamp_list, ip_list], fw)


for _i in range(1, 6):
    save_file(_i)

