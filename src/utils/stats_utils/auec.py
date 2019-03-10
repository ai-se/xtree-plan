import numpy as np
from pdb import set_trace

def squash(x):
    return 1 / (1 + np.exp(-np.sum(x)))

def compute_auec(dframe, y_max, y_min):
    x = dframe[dframe.columns[0]].values
    y = dframe[dframe.columns[1]].values
    [x_max, _] = dframe.max(axis=0).values
    [x_min, _] = dframe.min(axis=0).values
    y_norm = y / np.sum(y)
    return int(100 * squash(y_norm))
