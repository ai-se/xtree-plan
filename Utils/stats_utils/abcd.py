from __future__ import division, print_function

import numpy as np
from sklearn.metrics import *


def abcd(actual, predicted, distribution, as_percent=True, threshold=0):
    actual = [1 if a > 0 else 0 for a in actual]
    """
    Confusion Matrix:

    |`````````````|`````````````|
    |  TN[0][0]   |  FP[0][1]   |
    |             |             |
    |`````````````|`````````````|
    |  FN[1][0]   |  TP[1][1]   |
    |             |             |
    `````````````````````````````
    """

    def stringify(lst):
        try:
            return [str(int(a)) for a in lst]
        except ValueError:
            return [str(a) for a in lst]

    predicted = [1 if val > 0.4 else 0 for val in distribution]
    auroc = round(roc_auc_score(actual, distribution), 2)
    c_mtx = confusion_matrix(actual, predicted)

    "Probablity of Detection: Pd"
    try:
        p_d = c_mtx[1][1] / (c_mtx[1][1] + c_mtx[1][0])  # TP/(TP+FN)
    except:
        p_d = 0

    "Probability of False Alarm: Pf"
    try:
        p_f = c_mtx[0][1] / (c_mtx[0][1] + c_mtx[0][0])  # FP/(FP+TN)
    except:
        p_f = 0

    "Precision"
    try:
        p_r = c_mtx[1][1] / (c_mtx[1][1] + c_mtx[0][1])  # TP/(TP+FP)
        if not np.isfinite(p_r): p_r = 0
    except:
        p_r = 0

    "Recall (Same as Pd)"
    r_c = p_d

    "F1 measure"
    try:
        f1 = 2 * c_mtx[1][1] / (2 * c_mtx[1][1] + c_mtx[0][1] + 1 * c_mtx[1][
            0])  # F1 = 2*TP/(2*TP+FP+FN)
    except:
        f1 = 0

    ed = np.sqrt(0.7 * (1 - p_d) ** 2 + 0.3 * p_f ** 2)
    # e_d = 1 / ((0.5 / p_d) + (0.5 / (1 - p_f)))
    e_d = 2 * p_d * (1 - p_f) / (1 + p_d - p_f)
    g = np.sqrt(
        p_d - p_d * p_f)  # Harmonic Mean between True positive rate and True negative rate
    # set_trace()
    if np.isnan(p_d or p_f or p_r or r_c or f1 or e_d or g or auroc):
        return 0, 0, 0, 0, 0, 0, 0, 0
    if as_percent is True:
        return p_d * 100, p_f * 100, p_r * 100, r_c * 100, f1 * 100, e_d * 100, g * 100, auroc * 100
    else:
        return p_d, p_f, p_r, r_c, f1, e_d, g, auroc


def print_stats(actual, predicted, name="name", header=False):
    if header:
        print("Name", "PD  ", "PF  ", "Prec", "Rec ", "F1  ", "Bal ", "G   ",
              sep="\t")
    name = name[:4] if len(name) >= 4 else name + (4 - len(name)) * " "
    print(name,
          "{0:0.2f}\t{1:0.2f}\t{2:0.2f}\t{3:0.2f}\t{4:0.2f}\t{5:0.2f}\t{6:0.2f}".format(
              *abcd(actual, predicted)),
          sep="\t")


def _test_abcd():
    x = [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0]
    y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1]
    stats = abcd(x, y)
    print_stats(x, y)


if __name__ == "__main__":
    _test_abcd()
