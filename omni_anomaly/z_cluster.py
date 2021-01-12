from scipy.stats import wasserstein_distance
import pickle
import os
import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy


def compute_two_dataset(m1, m2):
    assert m1.shape[1] == 3
    assert m2.shape[1] == 3
    dis = 0
    for i in range(3):
        dis += wasserstein_distance(m1[:, i], m2[:, i])
    return dis


def run_compute_two_dataset(args):
    return compute_two_dataset(*args)


def compute_z_matrix(machine_list, result_dir='result_for_period1', save_z_dir='z_results',
                     distance_matrix_name='z_distance.pkl', machine_file_name='machine_list.pkl'):
    new_machine_list = []
    machine_data_dict = {}
    for m in machine_list:
        try:
            with open(os.path.join(result_dir, f'{m}-train_z.pkl'), 'rb') as f:
                machine_data_dict[m] = pickle.load(f)
            new_machine_list.append(m)
        except:
            print(f'machine {m} failed.')
    machine_number = len(new_machine_list)
    distance_matrix = np.zeros([machine_number, machine_number])
    args_list = []
    for i in range(machine_number):
        for j in range(i):
            args_list.append((
                machine_data_dict[new_machine_list[i]], machine_data_dict[new_machine_list[j]]
            ))
    pool = Pool(50)
    result_list = pool.map(run_compute_two_dataset, args_list)
    k = 0
    for i in range(machine_number):
        for j in range(i):
            distance_matrix[i, j] = result_list[k]
            distance_matrix[j, i] = distance_matrix[i, j]
            k += 1
    distance_matrix = np.round(distance_matrix, 2)
    os.makedirs(save_z_dir, exist_ok=True)
    with open(os.path.join(save_z_dir, distance_matrix_name), 'wb') as f:
        pickle.dump(distance_matrix, f)
    with open(os.path.join(save_z_dir, machine_file_name), 'wb') as f:
        pickle.dump(new_machine_list, f)


def cluster_on_z_matrix(save_z_dir='z_results', distance_matrix_name='z_distance.pkl', N=5, png_filename='HAC.png',
                        cluster_filename='z_cluster.pkl'):
    with open(os.path.join(save_z_dir, distance_matrix_name), 'rb') as f:
        z_distance_matrix = pickle.load(f)
    total_number = z_distance_matrix.shape[0]
    ytdist = np.hstack([z_distance_matrix[j, j+1:] for j in range(total_number)])
    Z = hierarchy.linkage(ytdist, 'ward')  # 'average', 'centroid',
    plt.figure()
    dn = hierarchy.dendrogram(Z)
    plt.title('HAC algorithm under ward distance')
    plt.savefig(os.path.join(save_z_dir, png_filename))
    # plt.show()

    Z = Z[:, :2]
    class_number = np.unique(Z[-N+1:, :])
    class_number = class_number[class_number < total_number + Z.shape[0] - N + 1]
    result_list = []
    for each_class in class_number:
        if each_class < total_number:
            number_list = [each_class]
        else:
            number_list = Z[int(each_class - total_number), :]
            while np.max(number_list) >= total_number:
                number_list_old = number_list[number_list < total_number]
                number_list_new = number_list[number_list >= total_number]
                number_list_new = np.hstack([Z[int(i - total_number), :] for i in number_list_new])
                number_list = np.hstack([number_list_new, number_list_old])
        result_list.append(number_list)
    print(f'clustering result {N} classes: {result_list}')
    with open(os.path.join(save_z_dir, cluster_filename), 'wb') as f:
        pickle.dump(result_list, f)


