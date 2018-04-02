from __future__ import division

import os
import sys
from pdb import set_trace
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from Utils.FileUtil import list2dataframe
from smote import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

root = os.path.join(os.getcwd().split('src')[0], 'src')
if root not in sys.path:
    sys.path.append(root)


def getTunings(fname):
    raw = pd.read_csv(root + '/old/tunings.csv').transpose().values.tolist()
    formatd = pd.DataFrame(raw[1:], columns=raw[0])
    try:
        return formatd[fname].values.tolist()
    except KeyError:
        return None


def rforest(train, target):
    clf = RandomForestClassifier(n_estimators=100, random_state=1)
    try:
        source = list2dataframe(train)
    except IOError:
        source = train

    # source = SMOTE(source)

    source.loc[source[source.columns[-1]] == 0, source.columns[-1]] = False
    features = source.columns[:-1]
    klass = list(source[source.columns[-1]])
    clf.fit(source[features], klass)
    preds = clf.predict(target[target.columns[:-1]])
    distr = clf.predict_proba(target[target.columns[:-1]])[:, 1]

    return preds, distr


def rforest_grid_tuned(train, target):
    clf = RandomForestClassifier(n_estimators=800,
                                 max_depth=6,
                                 min_samples_leaf=6,
                                 max_features=0.33)
    try:
        source = list2dataframe(train)
    except IOError:
        source = train

    source = SMOTE(source)

    # use a full grid over all parameters
    param_grid = {"max_depth": [3, None],
                  "max_features": [1, 3, 10],
                  "min_samples_split": [1, 3, 10],
                  "min_samples_leaf": [1, 3, 10],
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]}

    source.loc[source[source.columns[-1]] == 0, source.columns[-1]] = False
    features = source.columns[:-1]
    klass = list(source[source.columns[-1]])
    clf = GridSearchCV(clf, param_grid).fit(source[features], klass)
    preds = clf.predict(target[target.columns[:-1]])
    distr = clf.predict_proba(target[target.columns[:-1]])[:, 1]

    return preds, distr


def xgboost(train, target):
    try:
        source = list2dataframe(train)
    except IOError:
        source = train

    # source = SMOTE(source)

    clf = GradientBoostingClassifier(n_estimators=80,
                                     max_depth=6,
                                     min_samples_leaf=6,
                                     learning_rate=0.085,
                                     subsample=True,
                                     max_features=0.33)

    source.loc[source[source.columns[-1]] == 0, source.columns[-1]] = False
    features = source.columns[:-1]
    klass = list(source[source.columns[-1]])
    clf.fit(source[features], klass)
    preds = clf.predict(target[target.columns[:-1]])
    distr = clf.predict_proba(target[target.columns[:-1]])[:, 1]
    preds = [1 if val > 0.77 else 0 for val in distr]
    return preds, distr


def xgboost_grid_tuned(train, target):
    try:
        source = list2dataframe(train)
    except IOError:
        source = train

    source = SMOTE(source)

    # Tune with grid search

    param_grid = {
        "n_estimators": [80],#, 40, 20],
        "learning_rate": [0.1],
        # "max_depth": [4, 6],
        # "min_samples_leaf": [3, 5, 9, 17],
        # "max_features": [1.0, 0.3, 0.1]
    }

    clf = GradientBoostingClassifier()
    source.loc[source[source.columns[-1]] == 0, source.columns[-1]] = False
    features = source.columns[:-1]
    klass = list(source[source.columns[-1]])
    clf = GridSearchCV(clf, param_grid).fit(source[features], klass)
    preds = clf.predict(target[target.columns[:-1]])
    distr = clf.predict_proba(target[target.columns[:-1]])[:, 1]

    return preds, distr


def _test_model():
    pass


if __name__ == '__main__':
    _test_model()
