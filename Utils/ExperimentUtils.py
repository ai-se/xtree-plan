from __future__ import print_function, division

import os
import sys

# Update path
root = os.path.join(os.getcwd().split('src')[0], 'src')
if root not in sys.path:
    sys.path.append(root)

import pandas as pd
from random import uniform
from Utils.StatsUtils.ABCD import abcd
from random import uniform as random
from pdb import set_trace
import numpy as np

def overlaps(new, validation):
    """
    Have the changes been implemented?"
    """

    "Create a smaller dframe of all non-defective modules in validation set"
    closed_in_validation = validation[validation['bugs'].isin([0])]

    "Group the smaller dframe and the patched dframe by their file names"
    modules = list(set(closed_in_validation["name"].tolist()))

    heeded = []
    for module_name in modules:
        count = []
        module_name_new = new[new["name"].isin([module_name])]
        module_name_val = closed_in_validation[closed_in_validation["name"].isin([module_name])]
        for col_name in module_name_val.columns[1:-1]:
            aa = module_name_new[col_name]
            bb = module_name_val[col_name]
            try:
                ranges = sorted(eval(aa.values.tolist()[0]))
                count.append(any([abs(ranges[0]) <= bbb <= abs(ranges[1]) for bbb in bb.tolist()]))
            except TypeError:
                count.append(any([bbb == aa.values[0] for bbb in bb.tolist()]))
            except IndexError:
                pass
        if len(count) > 0:
            heeded.append(sum(count)/len(count))
    
    return heeded

def impact(test, pred):
    actuals = test[test.columns[-1]]
    gain = int(100 * (1 - sum(pred) / sum(actuals)))
    return gain


def pred_stats(before, after, distr):
    pd, pf = abcd(before, after, distr)[:2]
    return round(pd, 2), round(pf, 2)


def apply(changes, row):
    all = []
    for idx, thres in enumerate(changes):
        newRow = row
        if thres > 0:
            if newRow[idx] > thres:
                newRow[idx] = uniform(0, thres)
            all.append(newRow)

    return all


def apply2(changes, row):
    new_row = row
    for idx, thres in enumerate(changes):
        if thres is not None:
            if new_row[idx] > thres:
                new_row[idx] = uniform(0, thres)

    # delta = np.array(new_row) - np.array(row)
    # delta_bool = [1 if a > 0 else -1 if a < 0 else 0 for a in delta]
    return new_row


def apply3(row, cols, pk_best):
    newRow = row
    for idx, col in enumerate(cols):
        try:
            proba = pk_best[col][0]
            thres = pk_best[col][1]
            if thres is not None:
                if newRow[idx] > thres:
                    newRow[idx] = uniform(0, thres) if random(0, 100) < proba else \
                        newRow[idx]
        except:
            pass

    return newRow


def deltas(orig, patched):
    delt_numr = []
    delt_bool = []
    for row_a, row_b in zip(orig.values[:-1], patched.values[:-1]):
        delt_bool.append([1 if a != b else 0 for a, b in zip(row_a, row_b)])

    delt_bool = np.array(delt_bool)
    fractional_change = np.sum(delt_bool, axis=0) * 100 / len(delt_bool)

    return fractional_change.tolist()


def deltas_count(columns, changes):
    delt_numr = {c:0 for c in columns}
    for change in changes:
        for key, val in change.iteritems():
            try:
                delt_numr[key] += abs(val)
            except KeyError:
                delt_numr.update({key: val})
    delta = [delt_numr[key]*100/len(changes) for key in columns]
    return delta

def deltas_magnitude(columns, changes):
    delt_numr_pos = {c:0 for c in columns}
    delt_numr_neg = {c:0 for c in columns}
    for change in changes:
        for key, val in change.iteritems():
            try:
                if val > 0:
                    delt_numr_pos[key] += val
                elif val<0:
                    delt_numr_neg[key] += val
            except KeyError:
                if val>0:
                    delt_numr_pos.update({key: val})
                elif val<0:
                    delt_numr_neg.update({key: val})


    delta_pos = [delt_numr_pos[key]*100/len(changes) for key in columns]
    delta_neg = [delt_numr_neg[key]*100/len(changes) for key in columns]
    return delta_pos, delta_neg


# def deltas_magnitude(orig, patched):
#     delt_pos = []
#     delt_neg = []
#     for row_a, row_b in zip(orig.values[:-1], patched.values[:-1]):
#         delt_pos.append([1 if a>b else 0 for a, b in zip(row_a, row_b)])
#         delt_neg.append([-1 if a<b else 0 for a, b in zip(row_a, row_b)])
#
#     return pd.DataFrame(delt_pos, columns=orig.columns)\
#         , pd.DataFrame(delt_neg, columns=orig.columns)


class Changes():
    """
    Record changes.
    """

    def __init__(self):
        self.log = {}

    def save(self, name=None, old=None, new=None):
        if old > new:
            delt = -1
        elif old < new:
            delt = +1
        else:
            delt = 0
        self.log.update({name: delt})
