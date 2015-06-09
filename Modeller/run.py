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


class XOMO():

  def __init__(i):
    i.header = Xomo(model="all").names
    pass

  def toCSV(i, data, dir='Data/', name=None):
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


def predictor(tbl):
  rows = [r.cells for r in tbl._rows]
  effort = []
  for elem in rows:
    #     for _ in xrange(10):
    #       print(XOMO().model(elem[:-2]))
    try:
      effort += [howMuchEffort({h.name: el for h,
                                el in zip(tbl.headers[:-2],
                                          elem[:-2])})]
    except:
      try:
        effort += [howMuchEffort({h.name[1:]: el for h,
                                  el in zip(tbl.headers[:-2],
                                            elem[:-2])})]
      except:
        pass
#     print(elem[-2], effort[-1])
  return effort


def learner(lst=[], reps=24):
  mdl = XOMO()
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


if __name__ == "__main__":
  lst = learner(reps=10)
  try:
    rdivDemo(lst, isLatex=True)
  except:
    set_trace()
  #------- Debug --------
  set_trace()
