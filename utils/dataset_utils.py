import pandas as pd
import csv
import numpy as np


def load_data(csv_file, column_range=None):
    data = pd.read_csv(csv_file)
    X = data.values
    cols_numeric_attr = list(range(column_range[0], column_range[1]+1))
    feat_matrix = X[:, cols_numeric_attr]
    return feat_matrix


def save_data(org_file, feat_wm, saved_file, column_range):
    org_data = pd.read_csv(org_file)
    df = org_data.copy(deep=True)
    df.iloc[:, column_range[0]:column_range[1]+1] = feat_wm
    df.to_csv(saved_file, index=False)
    # print(f'save watermarked dataset at {saved_file}')


def load_storedkey(storedkey_file):
    info_dict = {}

    with open(storedkey_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            key = row[0]
            val = row[1]

            # Convert the value to a number or array
            if key in ['ampt', 'theta', 'bin_width', 'resolution', 'max_abs', 'num_bins']:
                val = float(val)
            elif key in ['p_vec_x', 'p_vec_y']:
                val = np.fromstring(val[1:-1], sep=' ')

            info_dict[key] = val

    return info_dict


def load_bin_info(stored_bininfo_file):
    nums_dict = {}
    with open(stored_bininfo_file, 'r') as file:
        for line in file:
            key, value = line.strip().split(':')
            nums_dict[float(key.strip())] = int(value.strip())
    return nums_dict
