from __future__ import division
from pdb import set_trace
from os import environ, getcwd
import sys
from scipy.stats.mstats import mode
from scipy.spatial.distance import euclidean
from numpy import mean
# Update PYTHONPATH
HOME = environ['HOME']
axe = HOME + '/git/axe/axe/'  # AXE
pystat = HOME + '/git/pystats/'  # PySTAT
cwd = getcwd()  # Current Directory
sys.path.extend([axe, pystat, cwd])
from random import choice, uniform as rand
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from smote import *
import pandas as pd
from abcd import _Abcd
from methods1 import *
from sk import rdivDemo


def formatData(tbl):
  Rows = [i.cells for i in tbl._rows]
  headers = [i.name for i in tbl.headers]
  return pd.DataFrame(Rows, columns=headers)


def Bugs(tbl):
  cells = [i.cells[-2] for i in tbl._rows]
  return cells

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PREDICTION SYSTEMS:
# ```````````````````
# 1. WHERE2 2. RANDOM FORESTS, 3. DECISION TREES, 4. ADABOOST,
# 5. LOGISTIC REGRESSION
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def where2prd(train, test, tunings=[None, None], smoteit=False, thresh=1):
  "WHERE2"

  def flatten(x):
    """
    Takes an N times nested list of list like [[a,b],[c, [d, e]],[f]]
    and returns a single list [a,b,c,d,e,f]
    """
    result = []
    for el in x:
      if hasattr(el, "__iter__") and not isinstance(el, basestring):
        result.extend(flatten(el))
      else:
        result.append(el)
    return result

  def leaves(node):
    """
    Returns all terminal nodes.
    """
    L = []
    if len(node.kids) > 1:
      for l in node.kids:
        L.extend(leaves(l))
      return L
    elif len(node.kids) == 1:
      return [node.kids]
    else:
      return [node]

  train_DF = createTbl(
      train,
      settings=tunings[0],
      _smote=False,
      isBin=False,
      bugThres=2)
  test_df = createTbl(test)
  t = discreteNums(train_DF, map(lambda x: x.cells, train_DF._rows))
  myTree = tdiv(t, opt=tunings[1])
  testCase = test_df._rows
  rows, preds = [], []
  for tC in testCase:
    newRow = tC
    loc = drop(tC, myTree)  # Drop a test case in the tree & see where it lands
    leafNodes = flatten(leaves(loc))
    # set_trace()
    rows = [leaf.rows for leaf in leafNodes][0]
    vals = [r.cells[-2] for r in rows]
    preds.append(0 if mean([k for k in vals]).tolist() < thresh else 1)
    # if median(vals) > 0 else preds.extend([0])
  return preds


def _where2pred():
  "Test where2"
  dir = '../Data'
  one, two = explore(dir)
  # set_trace()
  # Training data
  train = one[0][:-1]
  # Test data
  test = [one[0][-1]]
  actual = Bugs(createTbl(test, isBin=True))
  preds = where2prd(train, test)
  # for a, b in zip(actual, preds): print a, b
  # set_trace()
  return _Abcd(before=actual, after=preds, show=False)[-1]


def rforest(train, test, tunings=None, smoteit=True):
  "    RF"
  # Apply random forest Classifier to predict the number of bugs.
  if smoteit:
    train = SMOTE(train)
  if not tunings:
    clf = RandomForestClassifier(n_estimators=100)
  else:
    clf = RandomForestClassifier(n_estimators=int(tunings[0]),
                                 max_features=tunings[1] / 100,
                                 min_samples_leaf=int(tunings[2]),
                                 min_samples_split=int(tunings[3])
                                 )
  train_DF = formatData(train)
  test_DF = formatData(test)
  features = train_DF.columns[:-2]
  klass = train_DF[train_DF.columns[-2]]
  # set_trace()
  clf.fit(train_DF[features], klass)
  preds = clf.predict(test_DF[test_DF.columns[:-2]]).tolist()
  return preds


def _RF():
  "Test RF"
  dir = '../Data'
  one, two = explore(dir)
  # Training data
  train_DF = createTbl([one[0][0]])
  # Test data
  test_df = createTbl([one[0][1]])
  actual = Bugs(test_df)
  preds = rforest(train_DF, test_df, mss=6, msl=8,
                  max_feat=4, n_est=5756,
                  smoteit=False)
  print _Abcd(before=actual, after=preds, show=False)[-1]


def CART(train, test, tunings=None, smoteit=True, duplicate=True):
  "  CART"
  # Apply random forest Classifier to predict the number of bugs.
  if smoteit:
    train = SMOTE(train, atleast=50, atmost=101, resample=duplicate)

  if not tunings:
    clf = DecisionTreeClassifier(criterion='entropy',)
  else:
    clf = DecisionTreeClassifier(max_depth=int(tunings[0]),
                                 min_samples_split=int(tunings[1]),
                                 min_samples_leaf=int(tunings[2]),
                                 max_features=float(tunings[3] / 100),
                                 max_leaf_nodes=int(tunings[4]),
                                 criterion='entropy')
  train_DF = formatData(train)
  test_DF = formatData(test)
  features = train_DF.columns[:-2]
  klass = train_DF[train_DF.columns[-2]]
  # set_trace()
  clf.fit(train_DF[features].astype('float32'), klass.astype('float32'))
  preds = clf.predict(test_DF[test_DF.columns[:-2]].astype('float32')).tolist()
  return preds


def _CART():
  "Test CART"
  dir = './Data'
  one, two = explore(dir)
  # Training data
  train_DF = createTbl(one[0])
  # Test data
  test_df = createTbl(two[0])
  actual = Bugs(test_df)
  preds = CART(train_DF, test_df)
  set_trace()
  _Abcd(train=actual, test=preds, verbose=True)


def adaboost(train, test, smoteit=True):
  "ADABOOST"
  if smoteit:
    train = SMOTE(train)
  clf = AdaBoostClassifier()
  train_DF = formatData(train)
  test_DF = formatData(test)
  features = train_DF.columns[:-2]
  klass = train_DF[train_DF.columns[-2]]
  # set_trace()
  clf.fit(train_DF[features], klass)
  preds = clf.predict(test_DF[test_DF.columns[:-2]]).tolist()
  return preds


def _adaboost():
  "Test AdaBoost"
  dir = './Data'
  one, two = explore(dir)
  # Training data
  train_DF = createTbl(one[0])
  # Test data
  test_df = createTbl(two[0])
  actual = Bugs(test_df)
  preds = adaboost(train_DF, test_df)
  set_trace()
  _Abcd(train=actual, test=preds, verbose=True)


def logit(train, test, smoteit=True):
  "Logistic Regression"
  if smoteit:
    train = SMOTE(train)
  clf = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0,
                           fit_intercept=True, intercept_scaling=1,
                           class_weight=None, random_state=None)
  train_DF = formatData(train)
  test_DF = formatData(test)
  features = train_DF.columns[:-2]
  klass = train_DF[train_DF.columns[-2]]
  # set_trace()
  clf.fit(train_DF[features], klass)
  preds = clf.predict(test_DF[test_DF.columns[:-2]]).tolist()
  return preds


def _logit():
  "Test LOGIT"
  dir = './Data'
  one, two = explore(dir)
  # Training data
  train_DF = createTbl(one[0])
  # Test data
  test_df = createTbl(two[0])
  actual = Bugs(test_df)
  preds = logit(train_DF, test_df)
  set_trace()
  _Abcd(train=actual, test=preds, verbose=True)


def knn(train, test, smoteit=True):
  "kNN"
  if smoteit:
    train = SMOTE(train)
  neigh = KNeighborsClassifier()
  train_DF = formatData(train)
  test_DF = formatData(test)
  features = train_DF.columns[:-2]
  klass = train_DF[train_DF.columns[-2]]
  # set_trace()
  neigh.fit(train_DF[features], klass)
  preds = neigh.predict(test_DF[test_DF.columns[:-2]]).tolist()
  return preds

if __name__ == '__main__':
  random.seed(0)
  Dat = []
  for _ in xrange(10):
    print(_where2pred())
#  Dat.insert(0, 'Where2 untuned')
#  rdivDemo([Dat])
