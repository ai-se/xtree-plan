import os
import sys
import numpy
from pathlib import Path

from pdb import set_trace
root = Path(os.path.abspath(os.path.join(os.getcwd().split("src")[0], 'src')))
if root not in sys.path:
    sys.path.append(str(root))

import pandas as pd
import re

if __name__ == "__main__":
    cur_path = root.joinpath("results/RQ2/")
    projects = [d for d in cur_path.iterdir() if d.is_dir()]
    rq2_inc = []
    for proj in projects:
        for data in proj.glob("*_inc.csv"):
            dframe = pd.read_csv(data, index_col=False)
            # -- Sort the values --
            dframe = dframe.sort_values(['Overlap', 'Method'])
            # -- Group by mean --
            dframe = dframe.groupby(['Overlap', 'Method']).mean()
            # -- Covert to integer --
            dframe['Num'] = dframe['Num'].astype('int')
            df = pd.DataFrame(dframe.values.reshape(
                (1, -1)), columns=dframe.index)
            df.index = df.index[:-1].tolist() + [data.name[:-8]]
            rq2_inc.append(df)

    rq2_inc = pd.concat(rq2_inc)
    rq2_inc.to_csv('RQ2_inc.csv')
