import json
import os
import scipy.io
import numpy as np
from scipy import linalg
import pickle
import fnmatch
import time


def load_wm_params(json_file, key_idx, eigenvectors):
    with open(json_file) as json_f:
        info = json.load(json_f)
    params = info[key_idx]

    # Extract specific parameters
    key_id = params['key_id']
    ampt = params['ampt']
    theta = params['theta']
    bin_width = params['bin_width']
    resolution = params['resolution']
    p_vec_1_id = params['p_vecs']['1']
    p_vec_2_id = params['p_vecs']['2']

    wm_params = {
        'key_id': key_id,
        'ampt': ampt,
        'theta': theta,
        'bin_width': bin_width,
        'resolution': resolution,
        'p_vec_x': eigenvectors[p_vec_1_id],
        'p_vec_y': eigenvectors[p_vec_2_id]
    }

    return wm_params


def generate_e2(d):
    e2 = np.ones(d) / np.sqrt(d)
    return e2


def generate_basis(samples, e2):
    # Project samples onto the hyperplane defined by e2
    projections = samples - np.outer(np.dot(samples, e2), e2)

    A = projections.T
    C = np.dot(A, A.T)

    # Calculate eigenvalues and eigenvectors of C
    eigenvalues, eigenvectors = np.linalg.eigh(C)

    return eigenvalues, eigenvectors.T


def save_basis(basis, file_path):
    np.save(file_path, basis)


def load_eigenvectors(data, eigenvec_file):
    if os.path.exists(eigenvec_file):
        if fnmatch.fnmatch(eigenvec_file, "*.mat"):
            eigen_vecs_norm = scipy.io.loadmat(eigenvec_file)['eigenVectors']
        elif fnmatch.fnmatch(eigenvec_file, "*.pickle"):
            with open(eigenvec_file, 'rb') as f:
                eigen_vecs_norm = pickle.load(f)
        elif fnmatch.fnmatch(eigenvec_file, "*.npy"):
            eigen_vecs_norm = np.load(eigenvec_file)
        else:
            print('Please check the suffix of eigenvec_file')
            exit()
    else:
        d = data.shape[1]
        e2 = generate_e2(d)
        eigenvalues, eigen_vecs_norm = generate_basis(data, e2)
        # for idx, eigenvec in enumerate(eigen_vecs_norm):
        #     print(f'\nidx {idx}, eigenvalue is {eigenvalues[idx]}, eigenvector is\n{eigenvec}')
        save_basis(eigen_vecs_norm, eigenvec_file)
        print(f'No eigenvec_file found, generate and save it at {eigenvec_file}')
    return eigen_vecs_norm




