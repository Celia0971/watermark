# discard
from astropy.timeseries import LombScargle
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import json
import re

from utils import avg_unique, load_storedkey, plot_ls_periodogram


def get_del_col_idx_list_param_d(file_name, num_col_del):
    match = re.search(r'_emb(\d+)', file_name)
    if match:
        num_col_emb = int(match.group(1))
        del_col_idx = []
        # print(f'param ans d: num_col_embed {num_col_emb}, num_col_del {num_col_del}')
        for i in range(1, num_col_del+1):
            del_col_idx.append(num_col_emb - i)
        # print(f'del_col_idx {del_col_idx}')
        return del_col_idx


def get_del_col_name_feat_rank(json_file, num_col_del):
    # Load the sorted dictionary from the JSON file
    with open(json_file, 'r') as json_file:
        sorted_dict = json.load(json_file)

    # Get the last 'num_col_del' keys from the sorted dictionary
    last_keys = list(sorted_dict.keys())[-num_col_del:]

    return last_keys


def get_col_names(csv_file):
    df = pd.read_csv(csv_file)
    col_names = list(df.columns)
    if col_names[0] == 'EventId':
        col_names = col_names[1:]
    return col_names


def find_column_indices(column_names, columns_to_find):
    """
    Find the indices of given column names in a list of column names.

    Args:
    column_names (list): List of column names.
    columns_to_find (list): List of column names to find indices for.

    Returns:
    list: List of indices corresponding to the columns found.
    """
    indices = []
    for column in columns_to_find:
        if column in column_names:
            indices.append(column_names.index(column))
        else:
            indices.append(None)  # None for columns not found
    return indices


def delete_elements_by_indices(vector, indices_to_delete):
    """
    Delete elements from a vector based on the given indices without modifying the vector in place.

    Args:
    vector (list): The input vector.
    indices_to_delete (list): List of indices to delete from the vector.

    Returns:
    array: A new vector with elements removed at specified indices.
    """
    res = np.array([element for index, element in enumerate(vector) if index not in indices_to_delete])
    return res


def delete_last_elements(arr, a):
    if a >= len(arr):
        return np.array([])
    else:
        return arr[:-a]


def extract_wm_col_del_feat_rank(susp_file, stroedkey, fig_file, num_col_to_del, col_range,
                                 param_ans_d=False, probabilities=None, fmax=60, span=8, saveFig=False):
    """
    Given a susp_file and a storedkey, extract watermark from it, output result
    :param susp_file: the path to suspicious file
    :param dataset: 'covtype'
    :param stroedkey: key info, dict
    :param fig_file: the path to the figure of lomb scargle periodogram
    :param probabilities: probabilities = [0.001, 0.003, 0.05, 0.01, 0.1, 0.2]
    :param fmax: maximum frequency, fmax = 60, default
    :param span: zoomed-in figure span
    :param saveFig: whether save ls periodogram
    :return: extract result (the probability)
    """
    # todo ori_file should not be an input
    # 1. load suspicious dataset
    data = pd.read_csv(susp_file)
    X = data.values
    cols_numeric_attr = list(range(col_range[0], col_range[1] + 1 - num_col_to_del))
    feat_numeric = X[:, cols_numeric_attr]
    feat_numeric = feat_numeric.astype(float)

    # 2. load stored key
    key_info = load_storedkey(stroedkey)
    # stored_bininfo_file = stroedkey.replace("_keys.csv", "_bin_info.txt")
    # print('stored_bininfo_file', stored_bininfo_file)
    # bin_info = load_bin_info(stored_bininfo_file)

    e1 = delete_last_elements(np.array(key_info['p_vec_x']), num_col_to_del)
    e2 = delete_last_elements(np.array(key_info['p_vec_y']), num_col_to_del)

    # 3. projection
    time_seq_x = np.dot(feat_numeric, np.transpose(e1))
    time_seq_y = np.dot(feat_numeric, np.transpose(e2))
    time_seq = np.vstack((time_seq_x, time_seq_y)).T

    # 4. sort
    time_seq = time_seq[np.argsort(time_seq[:, 0])]
    # 5. binning
    bin_vals = np.floor(time_seq[:, 0] / key_info['bin_width'])
    bin_vals_dict = dict(Counter(bin_vals))
    num_bins = len(bin_vals_dict)
    # print('===== num of bins', num_bins)

    # == plot start
    # bins = list(bin_vals_dict.keys())
    # frequencies = list(bin_vals_dict.values())
    #
    # # Plotting the bar figure
    # plt.bar(bins, frequencies)
    # plt.xlabel('Bins')
    # plt.ylabel('Frequency')
    # plt.title(f'Histogram of Bins, num {num_bins}')
    #
    # figname = susp_file.rsplit('.', 1)[0].split('/')[-1]
    # plt.close()
    # == plot end

    time_seq[:, 0] = np.floor(time_seq[:, 0] / key_info['bin_width']) / key_info['resolution']
    # 6. average unique
    time_seq_prime = avg_unique(time_seq)

    # 7. compute the Lomb-Scargle periodogram
    x = time_seq_prime[:, 0]
    y = time_seq_prime[:, 1]
    if any(x < 0):
        x = x - np.min(x)  # shift x so that it starts at zero

    # 7.1 ls
    ls = LombScargle(x, y, nterms=1, normalization='psd')

    # 7.2 compute the required peak height to attain any given false alarm probability
    if probabilities is None:
        probabilities = [0.001, 0.003, 0.05, 0.01, 0.1, 0.2]
    peak_height_prob = ls.false_alarm_level(probabilities)
    # for i, prob in enumerate(probabilities):
    #     print('false alarm probability={:.1f}%, required peak height={:.4f}'.format(prob, peak_height_prob[i]))

    # 7.3 autopower()
    frequency, power = ls.autopower(minimum_frequency=0, maximum_frequency=fmax)
    # for ii, freq in enumerate(frequency):
    #     print('freq={:.2f}, power={:.2f}'.format(freq, power[ii]))

    # 7.4 power at theta
    power_at_theta = ls.power(key_info['theta'])
    # print('power at theta', power_at_theta)

    # 7.5 how significant is the peak that at theta?
    # significant_max = ls.false_alarm_probability(power.max(), method='baluev')
    # print('significant max', significant_max)
    false_alarm_prob_at_theta = ls.false_alarm_probability(power_at_theta, method='baluev')
    # print('false alarm probability of the peak at theta', false_alarm_prob_at_theta)

    # 7.6 output result
    result = 1 - false_alarm_prob_at_theta
    # print('result', result)

    # 7.7 plot periodogram
    if saveFig:
        plot_ls_periodogram(frequency, power, probabilities, peak_height_prob, span, key_info['theta'], fig_file)
    else:
        pass
    return result
