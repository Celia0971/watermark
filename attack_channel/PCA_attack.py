import numpy as np
import argparse
import fnmatch
import random
import os
from tqdm import tqdm
import pandas as pd


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_file", default="", type=str, help="path to clean file (not watermarked)")
    parser.add_argument("--wm_folder", default="", type=str, help="folder, watermarked dataset")
    parser.add_argument("--alpha_list", nargs='+', default=[5, 10, 15, 20], type=int, help="the dimension of reduction")
    parser.add_argument("--output_dir", default='', type=str, help="path to output folder")
    parser.add_argument('--column_range', type=lambda x: tuple(map(int, x.split(','))), default=(0, 100), help="numerical attributes")
    parser.add_argument("--dataset", default='covtype', help="'covtype', 'winequality', 'hive', 'bioresponse', 'higgs', 'gas'")
    parser.add_argument("--method", default='Ours', choices=['Ours', 'OBT', 'SF', 'NR', 'GAHSW', 'IP'])
    args = parser.parse_args()
    return args


def get_covar_mat(X):
    """
    Calculate the covariance matrix of the input dataset X.

    Parameters:
    X (numpy.ndarray): Input dataset of shape (N, D).

    Returns:
    numpy.ndarray: Covariance matrix of shape (D, D).
    """
    return np.dot(X.T, X)


def get_eigen_vec(C):
    """
    Calculate eigenvectors of the covariance matrix and sort them based on eigenvalues.

    Parameters:
    C (numpy.ndarray): Covariance matrix of shape (D, D).

    Returns:
    numpy.ndarray: Matrix of eigenvectors of shape (D, D).
    """
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    # Sort eigenvectors based on eigenvalues in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    # Normalize eigenvectors
    eigenvectors /= np.linalg.norm(eigenvectors, axis=0)
    return eigenvectors


def get_new_dataset(r, V, X):
    """
    Convert the dataset X to a lower dimensional dataset X' using PCA.

    Parameters:
    r (int): Number of variables of dimensional reduction.
    V (numpy.ndarray): Matrix of eigenvectors of shape (D, D).
    X (numpy.ndarray): Input dataset of shape (N, D).

    Returns:
    numpy.ndarray: Transformed dataset X' of shape (N, L), where L = D - r.
    """
    return np.dot(X, V[:, :-r])


def reconstruct(X_prime, V, r):
    """
    Reconstruct the dataset by dropping the last r columns.

    Parameters:
    X_prime (numpy.ndarray): Transformed dataset X' of shape (N, L).
    r (int): Number of variables dropped during transformation.

    Returns:
    numpy.ndarray: Reconstructed dataset X_hat of shape (N, D).
    """
    P = V[:, :-r].T
    return np.dot(X_prime, P)


def standardize_dataset(X):
    """
    Standardize the dataset by subtracting the mean and dividing by the standard deviation along each feature.

    Parameters:
    X (numpy.ndarray): Input dataset of shape (N, D).

    Returns:
    numpy.ndarray: Standardized dataset of shape (N, D).
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    standardized_X = (X - mean) / std
    return standardized_X


def test():
    # test
    N = 100  # Number of samples
    D = 5  # Dimensionality of each sample
    r = 2
    # Generate a random dummy dataset
    np.random.seed(0)  # For reproducibility
    X_ = np.random.rand(N, D)
    X = standardize_dataset(X_)
    print("Dummy dataset X:")
    print(X)

    C = get_covar_mat(X)
    V = get_eigen_vec(C)
    print(f"eigen vec {V}")
    X_prime = get_new_dataset(r, V, X)
    X_re = reconstruct(X_prime, V, r)

    diff = X - X_re
    print("Diff:")
    print(diff)


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


def pca_attack(csv_file, column_range, r, saved_file):
    X = load_data(csv_file, column_range)
    C = get_covar_mat(X)
    V = get_eigen_vec(C)
    X_prime = get_new_dataset(r, V, X)
    X_re = reconstruct(X_prime, V, r)
    save_data(csv_file, X_re, saved_file, column_range)


if __name__ == "__main__":
    # test()
    args = args_parse()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    file_list = []
    for file_name in os.listdir(args.wm_folder):
        if fnmatch.fnmatch(file_name, "*.csv"):
            file_list.append(os.path.join(args.wm_folder, file_name))
    if args.clean_file != "":
        file_list.append(args.clean_file)
    else:
        print("Attention, clean file is null")

    print(f"\nPCA attack. Processing ...")
    for each_file in tqdm(file_list):
        csv_file_name = os.path.basename(each_file).split('.')[0]
        for alpha in args.alpha_list:
            file_name = "{}_pca_pca_{:.1f}_0.00_{}.csv".format(csv_file_name, alpha, args.method)
            if not os.path.exists(os.path.join(args.output_dir, file_name)):
                pca_attack(each_file, args.column_range, alpha, os.path.join(args.output_dir, file_name))
            else:
                print(f'file {file_name} exists')