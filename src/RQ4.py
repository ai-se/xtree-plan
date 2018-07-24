"""
Compare Bellwether XTREEs with other threshold based learners.
"""

import os
import sys

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
from tabulate import tabulate


def research_question_4(verbose=True):
    """
    RQ4: How many changes do each of the planners propose?

    Parameters
    ----------
    verbose: Bool
        Pretty print #changes
    """

    data = get_all_projects() # Get all the projects
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
        deltas[deltas.columns[1:]] = ((df - df.min()) / (df.max() - df.min())*100).fillna(value=0).astype(int)
        deltas.set_index("Metrics")
        if verbose:
            print(proj.upper())
            print(len(proj)*"-")
            print tabulate(deltas, headers='keys', tablefmt='simple', showindex=False)
            print("\n"+46*"=")


if __name__ == "__main__":
    research_question_4()
