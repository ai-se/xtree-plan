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


def list2dataframe(lst, binarize=False):
    """
    Convert a list of paths to pandas dataframe
    """
    data = []
    try:
        for elem in lst:
            dframe_temp = read_csv(elem)
            if binarize:
                dframe_temp.loc[dframe_temp['<bug'] > 0, '<bug'] = 1
            data.append(dframe_temp)
    except:
        dframe_temp = read_csv(lst)
        if binarize:
            dframe_temp.loc[dframe_temp['<bug'] > 0, '<bug'] = 1
        return dframe_temp

    return concat(data, ignore_index=True)
