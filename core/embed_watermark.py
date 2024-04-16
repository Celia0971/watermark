import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from scipy import stats

from utils import plot_pie_chart, ans_sine_signal


def sort_arrays(X, Y):
    # Zip the two arrays together
    combined = list(zip(X, Y))

    # Sort the combined list based on the first array (X)
    combined.sort(key=lambda x: x[0])

    # Extract the sorted X and Y values
    sorted_X, sorted_Y = zip(*combined)

    return np.array(list(sorted_X)), np.array(list(sorted_Y))


def calculate_average(dictionary):
    # Check if the dictionary is empty
    if not dictionary:
        return None

    # Sum all the values in the dictionary
    total = sum(dictionary.values())

    # Divide the total by the number of keys
    average = total / len(dictionary)

    return average


def embed_watermark(data, wm_params, storedkey_file, save_bin_info=False):
    X_wm = data.dot(wm_params['p_vec_x'])
    X_wm_hash = X_wm.copy()

    X_wm_hash = np.asarray(X_wm_hash, dtype=np.float64)
    bin_vals = np.floor(X_wm_hash / wm_params['bin_width'])
    bin_vals_dict = dict(Counter(bin_vals))
    num_bins = len(bin_vals_dict)
    # print(f'average #points in each bin {calculate_average(bin_vals_dict)}')

    # record bin info
    if save_bin_info:
        bin_info_file = storedkey_file.replace("_keys.csv", "_bin_info.txt")
        with open(bin_info_file, 'w') as file:
            for key, value in bin_vals_dict.items():
                file.write(str(key) + ': ' + str(value) + '\n')

    pie_save_path = storedkey_file.replace("_keys.csv", "_bin_pie.png")
    plot_pie_chart(bin_vals_dict, pie_save_path)

    X_wm_hash= np.floor(X_wm_hash / wm_params['bin_width']) / wm_params['resolution']
    Y_wm = wm_params['ampt'] * np.sin(wm_params['theta'] * 2 * np.pi * X_wm_hash)

    sine_lsp_fig_pth = storedkey_file.replace("_keys.csv", "_sine.png")
    power_at_theta = ans_sine_signal(X_wm_hash, Y_wm, wm_params, sine_lsp_fig_pth)

    # sort
    # X_wm_hash, Y_wm = sort_arrays(X_wm_hash, Y_wm)

    # estimate periods
    # num_periods = estimate_period(X_wm_hash, Y_wm)
    # print("Estimated Period:", num_periods)

    mat2add_wm = np.tile(wm_params['p_vec_y'].reshape(1, -1), (data.shape[0], 1)) * Y_wm.reshape(-1, 1)

    max_abs_val = np.max(np.abs(mat2add_wm))
    wm_params['max_abs'] = max_abs_val
    wm_params['num_bins'] = num_bins
    wm_params['avg_points_bin'] = calculate_average(bin_vals_dict)
    wm_params['power_at_theta'] = power_at_theta
    df = pd.DataFrame.from_dict(wm_params, orient='index', columns=['value'])
    df.to_csv(storedkey_file, header=False)
    # print(f'save secret key at {storedkey_file}')

    data_wm = data + mat2add_wm

    return data_wm
    
