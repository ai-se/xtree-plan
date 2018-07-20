"""
Compare Bellwether XTREEs with other threshold based learners.
"""

import os
import sys
from pdb import set_trace

# Update path
root = os.path.join(os.getcwd().split('src')[0], 'src')
if root not in sys.path:
    sys.path.append(root)

import pandas as pd
from planners.XTREE import xtree
from planners.alves import alves
from planners.shatnawi import shatnawi
from planners.oliveira import oliveira
from data.get_data import get_all_projects
from utils.file_util import list2dataframe
from pdb import set_trace

import random

TAXI_CAB = 1729
random.seed(TAXI_CAB)


def research_question_6(verbose=True):
    """
    How many changes do each of the planners propose

    """
    data = get_all_projects() # Get all the projects
    bellw = data.pop('lucene').data  # Bellwether dataset
    bellw = list2dataframe(bellw)
    all_proj = []
    for proj, paths in data.items():
        i = 0
        all_deltas = []
        for train, test in zip(paths.data[:-1], paths.data[1:]):
            i += 1

            "Convert to pandas type dataframe"
            train = list2dataframe(train)
            test = list2dataframe(test)

            "Create a DataFrame to hold the changes"
            deltas = pd.DataFrame()
            deltas["Metrics"] = test.columns[1:-1]
            deltas.set_index("Metrics")

            "Generate Patches"
            patched_xtree = xtree(train[train.columns[1:]], test)
            patched_alves = alves(train[train.columns[1:]], test)
            patched_shatw = shatnawi(train[train.columns[1:]], test)
            patched_olive = oliveira(train[train.columns[1:]], test)

            "Count Deltas"
            deltas["XTREE"] = test.ne(patched_xtree).sum().values[1:-1]
            deltas["Alves"] = test.ne(patched_alves).sum().values[1:-1]
            deltas["Shatw"] = test.ne(patched_shatw).sum().values[1:-1]
            deltas["Olive"] = test.ne(patched_olive).sum().values[1:-1]

            "Gather changes"
            all_deltas.append(deltas)

        for d in all_deltas[:-1]:
            deltas = deltas.add(d)

        deltas["Metrics"] = test.columns[1:-1]
        deltas.set_index("Metrics")
        "Normalize"
        df = deltas[deltas.columns[1:]]
        deltas[deltas.columns[1:]] = ((df - df.min()) / (
                        df.max() - df.min())*100).fillna(value=0).astype(int)
        all_proj.append(deltas)

    all_proj = pd.concat(all_proj, axis=1)
    all_proj.to_csv(os.path.abspath("results/RQ6/{}.csv".format(proj)), index=False)

if __name__ == "__main__":
    planning()
