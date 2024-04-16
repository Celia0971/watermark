import random
import pandas as pd
import numpy as np


def set_random_seed(seed):
    # Set random seed for pandas
    pd.options.mode.chained_assignment = None  # Disable pandas' chain warning
    pd.random.seed(seed)

    # Set random seed for numpy
    np.random.seed(seed)

    # Set random seed for the built-in random module
    random.seed(seed)
