#! /Users/rkrsn/miniconda/bin/python
from __future__ import print_function, division
from os import environ, getcwd
import sys

# Update PYTHONPATH
HOME = environ['HOME']
axe = HOME + '/git/axe/axe/'  # AXE
pystat = HOME + '/git/pystat/'  # PySTAT
SOURCE = '../SOURCE/'
sys.path.extend([axe, pystat, SOURCE])

import csv
from Prediction import *
from _imports import *
from abcd import _Abcd
from methods1 import *
import numpy as np
import pandas as pd
from pdb import set_trace
from WHAT import treatments as WHAT
from CROSSTREES import treatments as xtrees
from _model import xomod, howMuchEffort
from _XOMO import *
from numpy import sum, array
from sk import rdivDemo
from pom3.pom3 import pom3
from sklearn.ensemble import RandomForestRegressor
import random


def formatData(tbl, rows=None):
  if not rows:
    rows = [i.cells for i in tbl._rows]
  headers = [i.name for i in tbl.headers[:-1]]
  return pd.DataFrame(rows, columns=headers)


class RandomForest():

  def __init__(
          self,
          train=None,
          test=None,
          tuning=None,
          smoteit=False,
          duplicate=False):
    self.train = train
    self.test = test
    self.tuning = tuning
    self.smoteit = smoteit
    self.duplicate = duplicate

  def regress(self):
    "  RF"
    # Apply random forest Classifier to predict the number of bugs.
    if not self.tuning:
      clf = RandomForestRegressor(random_state=1)
    else:
      clf = RandomForestRegressor(n_estimators=int(tunings[0]),
                                  max_features=tunings[1] / 100,
                                  min_samples_leaf=int(tunings[2]),
                                  min_samples_split=int(tunings[3]),
                                  random_state=1)
    features = self.train.columns[:-1]
    klass = self.train[self.train.columns[-1]]
    # set_trace()
    clf.fit(self.train[features].astype('float32'), klass.astype('float32'))
    preds = clf.predict(
        self.test[self.test.columns[:-1]].astype('float32'))
    return preds


class XOMO():

  def __init__(i):
    i.header = Xomo(model="all").names
    pass

  def toCSV(i, data, dir='Data/XOMO/', name=None):
    with open(dir + name, 'w') as csvfile:
      writer = csv.writer(csvfile, delimiter=',',
                          quotechar='|', quoting=csv.QUOTE_MINIMAL)
      writer.writerow(['$' + d for d in i.header] + ['$>effort'])  # Header
      for cells in data[1]:  # Body
        indep = cells[:-4]
        depen = i.model(indep)
        body = indep + [depen]
        writer.writerow(body)
    return dir + name

  def genData(i, N=100):
    train = i.toCSV(xomod(N), name='Train.csv')
    test = i.toCSV(xomod(N), name='Test.csv')
    return createTbl([train]), createTbl([test])

  def model(i, X):
    row = {h: el for h, el in zip(i.header, X)}
    return howMuchEffort(row)


class POM3():

  def __init__(p3):
    p3.indep = ["Culture", "Criticality", "CriticalityModifier",
                "InitialKnown", "InterDependency", "Dynamism",
                "Size", "Plan", "TeamSize"]
    p3.depen = ['-cost', '+completion', '-idle']

  def toCSV(p3, data, dir='Data/POM3/', name=None):
    with open(dir + name, 'w') as csvfile:
      writer = csv.writer(csvfile, delimiter=',',
                          quotechar='|', quoting=csv.QUOTE_MINIMAL)
      writer.writerow(['$' + d for d in p3.indep] + [p3.depen[0]])  # Header
      for cells in data[1]:  # Body
        indep = cells[:-3]
        depen = p3.model(indep)
        body = indep + [depen]
        writer.writerow(body)
    return dir + name

  def genData(p3, N=100):
    train = p3.toCSV(pom3d(N), name='Train.csv')
    test = p3.toCSV(pom3d(N), name='Test.csv')
    return createTbl([train]), createTbl([test])

  def model(p3, X):
    try:
      return pom3().simulate(X)[0]
    except:
      set_trace()


<<<<<<< HEAD
def predictor(tbl, Model=XOMO):
  rows = [r.cells for r in tbl._rows]
  out = []
  for elem in rows:
    if Model == XOMO:
      out += [Model().model(elem[:-2])]
    elif Model == POM3:
      out += [Model().simulate(elem[:-2])[0]]
  return out
=======
def predictor(tbl=None, rows=None):
  if not rows:
    rows = [r.cells[:-2] for r in tbl._rows]
  effort = []
  for elem in rows:
    effort += [pom3().simulate(elem)[0]]
  return effort
>>>>>>> 50f375e1cba1d2ebde92849b97b14342b5523a32


def learner(mdl=XOMO, lst=[], reps=24):
  train, test = mdl().genData(N=1000)
  before = array(predictor(Model=mdl, tbl=train))
  for ext in [0, 0.25, 0.5, 0.75]:
    for info, prune in zip([0.25, 0.5, 0.75, 1.00],
                           [True, True, True, False]):
      prefix = '        Base' if ext == 0 else 'F=%0.2f, B=%0.2f' % (ext, info)
#       print(prefix)
      E = [prefix]
      for _ in xrange(reps):
        newTab = xtrees(train=train, test=test, verbose=False).main()
        after = array(predictor(tbl=newTab))
        E.append(sum(after) / sum(before))
      lst.append(E)
  return lst

<<<<<<< HEAD
=======

def dancer(mdl=POM3(), newRows=[], extent=0.1, what='MSE'):
  train1, _ = mdl.genData()
  train, _ = mdl.genData()

  def mutator(row):
    fact = lambda: random.choice([-1, 1])
    return [el * (1 + fact() * extent) for el in row]

  trainRows = [r.cells[:-1] for r in train1._rows]
  oldRows = [r.cells[:-1] for r in train._rows]
  trainDF = formatData(tbl=train, rows=trainRows)
  for row in oldRows[:-1]:
    newRows.append(mutator(row))
  testDF = formatData(tbl=train, rows=newRows)
  actual = array(predictor(rows=newRows))
  predicted = RandomForest(train=trainDF, test=testDF).regress()

  MRE = (actual - predicted) / actual * 100
  y = np.median(MRE, axis=0)
  yhi, ylo = np.percentile(MRE, q=[75, 25], axis=0)

  MSE = np.sqrt(sum((actual - predicted) ** 2)) / len(actual)
  if what == 'MRE':
    return y, yhi, ylo
  elif what == 'MSE':
    return MSE
  # --------- DEBUG ---------
#   set_trace()


>>>>>>> 50f375e1cba1d2ebde92849b97b14342b5523a32
if __name__ == "__main__":
  random.seed(0)
  for ext in np.linspace(0, 0.2, 6):
    d = dancer(extent=ext)
    print(ext, d)

#
#   lst = learner(reps=10)
#   try:
#     rdivDemo(lst, isLatex=True)
#   except:
#     set_trace()
# ------- Debug --------
#   set_trace()
