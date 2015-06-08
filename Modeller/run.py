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


class model():

  def __init__(i):
    pass

  def toCSV(i, data, dir='Data/', name=None):
    with open(dir + name, 'w') as csvfile:
      writer = csv.writer(csvfile, delimiter=',',
                          quotechar='|', quoting=csv.QUOTE_MINIMAL)
      writer.writerow(data[0][:-3])  # Header
      for cells in data[1]:  # Body
        indep = cells[:-4]
        depen = howMuchEffort({h[1:]: el for h,
                               el in zip(data[0][:-4], indep)})
        body = indep + [depen]
        writer.writerow(body)
    return dir + name

  def genData(i):
    train = i.toCSV(xomod(N=100), name='Train.csv')
    test = i.toCSV(xomod(N=100), name='Test.csv')
    return createTbl([train]), createTbl([test])

#   def XOMO(i, x):
#     "XOMO"
#     return (restructure(x))


def predictor(tbl):
  rows = [r.cells for r in tbl._rows]
  effort = []
  for elem in rows[:1]:
    for _ in xrange(10):
      print(howMuchEffort({h.name[1:]: el for h,
                           el in zip(tbl.headers[:-2], elem[:-2])}))
#     effort += [howMuchEffort({h.name[1:]: el for h,
#                               el in zip(tbl.headers[:-2], elem[:-2])})]
    print(elem[-2], effort[-1])
  set_trace()
  return effort


def learner():
  mdl = model()
  train, test = mdl.genData()
  before = predictor(tbl=train)
#           set_trace()
  newTab = WHAT(
      train=None,
      test=None,
      train_df=train,
      bin=False,
      test_df=test,
      extent=0.75,
      fSelect=True,
      far=False,
      infoPrune=0.5,
      method='best',
      Prune=True).main()
  after = predictor(tbl=newTab)
  return before, after


if __name__ == "__main__":
  (before, after) = learner()
  #------- Debug --------
  set_trace()
