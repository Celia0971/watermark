import numpy as np

def avg_unique(A):
    unique_A, index_A = np.unique(A[:, 0], return_inverse=True)
    B = np.column_stack((unique_A, np.bincount(index_A, weights=A[:, 1]) / np.bincount(index_A)))
    return B