from __future__ import print_function, division

import os
import sys
import re
# Update PYTHONPATH
root = os.path.join(os.getcwd().split('src')[0], 'src')
if root not in sys.path:
    sys.path.append(root)

from pdb import set_trace
from pandas import read_csv, concat
from pandas.io.common import EmptyDataError
from AxeUtils.w2 import where2, prepare, leaves
from AxeUtils.MakeAModel import MakeAModel


def new_table(tbl, headerLabel, Rows):
    tbl2 = clone(tbl)
    newHead = Sym()
    newHead.col = len(tbl.headers)
    newHead.name = headerLabel
    tbl2.headers = tbl.headers + [newHead]
    return clone(tbl2, rows=Rows)


def list2dataframe(lst, reshape=True):
    """
    Convert a list of paths to pandas dataframe
    """
    data = []
    try:
        for elem in lst:
            dframe_temp = read_csv(elem)
            if reshape:
                dframe_temp = dframe_temp[dframe_temp.columns[2:]]
                columns = dframe_temp.columns()
                columns = [re.sub("[^a-zA-Z0-9]", "", c) for c in columns]
                dframe_temp.columns = columns
            data.append(dframe_temp)
    except:
        dframe_temp = read_csv(lst)
        if reshape:
            dframe_temp = dframe_temp[dframe_temp.columns[2:]]
            columns = dframe_temp.columns
            columns = [re.sub("[^a-zA-Z0-9]", "", c) for c in columns]
            dframe_temp.columns = columns
        return dframe_temp

    return concat(data, ignore_index=True)


def create_tbl(
        data,
        settings=None,
        _smote=False,
        isBin=False,
        bugThres=1,
        duplicate=False):
    """
    kwargs:
    _smote = True/False : SMOTE input data.dat (or not)
    _isBin = True/False : Reduce bugs to defects/no defects
    _bugThres = int : Threshold for marking stuff as defective,
                      default = 1. Not defective => Bugs < 1
    """
    model = MakeAModel()
    _r = []
    for t in data:
        m = model.csv2py(t, _smote=_smote, duplicate=duplicate)
        _r += m._rows
    m._rows = _r
    # Initialize all parameters for where2 to run
    prepare(m, settings=None)
    tree = where2(m, m._rows)  # Decision tree using where2
    tbl = table(t)

    headerLabel = '=klass'
    Rows = []
    for k, _ in leaves(tree):  # for k, _ in leaves(tree):
        for j in k.val:
            tmp = j.cells
            if isBin:
                tmp[-1] = 0 if tmp[-1] < bugThres else 1
            tmp.append('_' + str(id(k) % 1000))
            j.__dict__.update({'cells': tmp})
            Rows.append(j.cells)

    return new_table(tbl, headerLabel, Rows)


def test_createTbl():
    dir = '../data.dat/camel/camel-1.6.csv'
    newTbl = create_tbl([dir], _smote=False)
    newTblSMOTE = create_tbl([dir], _smote=True)
    print(len(newTbl._rows), len(newTblSMOTE._rows))


def drop(test, tree):
    loc = apex(test, tree)
    return loc


if __name__ == '__main__':
    test_createTbl()
    set_trace()
