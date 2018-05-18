import os
import sys
from pdb import set_trace
import pandas as pd
import fnmatch


def recursive_glob(treeroot, pattern):
    results = []
    for base, dirs, files in os.walk(treeroot):
        goodfiles = fnmatch.filter(files, pattern)
        results.extend(os.path.join(base, f) for f in goodfiles)
    return results


def change_class_to_bool(fname):
    old = pd.read_csv(fname)
    # old.loc[old[old.columns[-1]] > 0, old.columns[-1]] = "T"
    # old.loc[old[old.columns[-1]] <= 0, old.columns[-1]] = "F"
    old[old.columns[2:]].to_csv(fname, index=False)


if __name__ == "__main__":
    res = recursive_glob(".", "*.csv")
    for fname in res:
        change_class_to_bool(fname)

    set_trace()
