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
from _model import xomod, howMuchEffort
from _XOMO import *
from numpy import sum, array
from sk import rdivDemo
from pom3.pom3 import pom3
import random


class XOMO():

  def __init__(i):
    i.header = Xomo(model="all").names
    pass

  def toCSV(i, data, dir='Data/XOMO/', name=None):
    with open(dir + name, 'w') as csvfile:
      writer = csv.writer(csvfile, delimiter=',',
                          quotechar='|', quoting=csv.QUOTE_MINIMAL)
      writer.writerow(['$' + d for d in i.header] + ['-effort'])  # Header
      for cells in data[1]:  # Body
        indep = cells[:-4]
        depen = i.model(indep)
        body = indep + [depen]
        writer.writerow(body)
    return dir + name

  def genData(i):
    train = i.toCSV(xomod(N=100), name='Train.csv')
    test = i.toCSV(xomod(N=100), name='Test.csv')
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

  def genData(p3):
    train = p3.toCSV(pom3d(N=100), name='Train.csv')
    test = p3.toCSV(pom3d(N=100), name='Test.csv')
    return createTbl([train]), createTbl([test])

  def model(p3, X):
    try:
      return pom3().simulate(X)[0]
    except:
      set_trace()


def predictor(tbl):
  rows = [r.cells for r in tbl._rows]
  effort = []
  for elem in rows:
    effort += [pom3().simulate(elem[:-2])[0]]
  return effort


def learner(mdl=POM3(), lst=[], reps=24):
  train, test = mdl.genData()
  before = array(predictor(tbl=train))
  for ext in [0, 0.25, 0.5, 0.75]:
    for info, prune in zip([0.25, 0.5, 0.75, 1.00],
                           [True, True, True, False]):
      prefix = '        Base' if ext == 0 else 'F=%0.2f, B=%0.2f' % (ext, info)
#       print(prefix)
      E = [prefix]
      for _ in xrange(reps):
        newTab = WHAT(
            train=None,
            test=None,
            train_df=train,
            bin=False,
            test_df=test,
            extent=ext,
            fSelect=True,
            far=False,
            infoPrune=info,
            method='best',
            Prune=prune).main()
        after = array(predictor(tbl=newTab))
        E.append(sum(after) / sum(before))
      lst.append(E)
  return lst


def dancer(mdl=POM3(), newRow=[], extent=0.1):
  train, _ = mdl.genData()
  actual = array(predictor(tbl=train))
  before = 
  def mutator(row):
    fact = lambda: random.choice([-1, 1])
    return [el * (1 + fact() * extent) for el in row]

  oldRows = [r.cells[:-2] for r in train._rows]

  for row in oldRows:
    newRow.append(mutator(row))


if __name__ == "__main__":
  random.seed(0)
  lst = learner(reps=10)
  try:
    rdivDemo(lst, isLatex=True)
  except:
    set_trace()
  #------- Debug --------
  set_trace()
