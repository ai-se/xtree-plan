"""
Compare Bellwether XTREEs with other threshold based learners.
"""

from __future__ import print_function, division

import os
import sys
from pdb import set_trace

# Update path
root = os.path.join(os.getcwd().split(
    'transfer-learning')[0], 'transfer-learning')
if root not in sys.path:
    sys.path.append(root)

import numpy as np
import pandas as pd
from data.get_data import get_all_projects
from Utils.FileUtil import list2dataframe
from commons.XTREE import xtree
from Utils.StatsUtils.CrossVal import TrainTestValidate
from random import random


import warnings
warnings.filterwarnings("ignore")


def effectiveness(dframe):
    overlap = dframe['Overlap']
    heeded = dframe['Heeded']

    a, b, c, d, e, f = 0, 0, 0, 0, 0, 0

    for over, heed in zip(overlap, heeded):
        if over < 50 and heed >= 0:
            a += 1
        if over > 50 and heed >= 0:
            b += 1
        if over < 50 and heed < 0:
            c += 1
        if over > 50 and heed < 0:
            d += 1
        if over == 0 and heed < 0:
            e += 1
        if over == 0 and heed >= 0:
            f += 1

    return a, b, c, d, e, f


def planning():
    data = get_all_projects()
    file = open(os.path.join(os.path.realpath(root), "results", "results.csv"), "w+")
    print("Name, Overlap<50 | Bugs Reduced, Overlap > 50 | Bugs Reduced, Overlap<50 | Bugs Increased , Overlap>50 | Bugs Increased, No Overlap | Bugs Increased, No Overlap | Bugs Decreased", file=file)
    for proj, paths in data.iteritems():
        # heeded = []
        i = 0
        for train, test, validation in TrainTestValidate.split(paths.data):
            i += 1
            results = pd.DataFrame()
            "Convert to pandas type dataframe"
            train = list2dataframe(train).dropna(axis=0).reset_index(drop=True)
            test = list2dataframe(test).dropna(axis=0).reset_index(drop=True)
            validation = list2dataframe(validation).dropna(
                axis=0).reset_index(drop=True)

            "Recommend changes with XTREE"
            new = xtree(train[train.columns[1:]], test)

            "Find modules that appear both in test and validation datasets"
            common_modules = pd.merge(
                test, validation, how='inner', on=['Name'])['Name']

            "Intitialize variables to hold information"
            improve_heeded = []
            overlap = []

            for module_name in common_modules:
                "For every common class, do the following ... "
                same = 0 # Keep track of features that follow XTREE's recommendations
                test_value = test.loc[test['Name'] == module_name] # Metric values of classes in the test set
                plan_value = new.loc[new['Name'] == module_name] # Metric values of classes in the XTREE's planned changes
                validate_value = validation.loc[validation['Name'] # Actual metric values the developer's changes yielded
                                                == module_name]
                

                for col in test_value.columns[1:-1]:
                    "For every metric (except the class name and the bugs), do the following ... "
                    try:
                        if isinstance(plan_value[col].values[0], str):
                            "If the change recommended by XTREE lie's in a range of values"
                            if eval(plan_value[col].values[0])[0] <= validate_value[col].values[0] <= eval(plan_value[col].values[0])[1]:
                                "If the actual change lies withing the recommended change, then increment the count of heeded values by 1"
                                same += 1
                        elif plan_value[col].values[0] == validate_value[col].values[0]:
                                "If the XTREE recommends no change and developers didn't change anything, that also counts as an overlap"
                                same += 1

                    except IndexError:
                        pass

                overlap.append(int(same/20 * 100)) # There are 20 metrics, so, find % of overlap for the class.
                heeded = test_value['bug'].values[0] - \
                    validate_value['bug'].values[0] # Find the change in the number of bugs between the test version and the validation version for that class.
                
                scale = -1 if heeded <= 0 else 1 # Used to add a random jiggle for the scatter plot.
                
                improve_heeded.append(0.5 * random() * scale + heeded) 

            "Save the results ... "

            results['Overlap'] = overlap
            results['Heeded'] = improve_heeded
            
            file = open(os.path.join(os.path.realpath(root), "results", "results.csv"), "a")
            print("{}_{},{},{},{},{},{},{}".format(
                proj, i, *effectiveness(results)), file=file)
            results.to_csv(os.path.join(os.path.realpath(root), "results", "{}/{}_{}.csv".format(proj, proj, i)), index=False)


if __name__ == "__main__":
    planning()
