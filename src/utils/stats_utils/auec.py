# from __future__ import print_function, division
from sklearn.metrics import auc

def compute_auec(dframe, y_max, y_min):
    x = dframe[dframe.columns[0]]
    y = dframe[dframe.columns[1]]
    [overlap_max, _] = dframe.max(axis=0).values
    [overlap_min, _] = dframe.min(axis=0).values
    area_norm = (overlap_max - overlap_min) * (y_max - y_min)
    auec_raw = auc(x, y, reorder=True)
    return round(auec_raw / area_norm, 2)

