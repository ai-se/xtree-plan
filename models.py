#! /Users/rkrsn/miniconda/bin/python
from __future__ import print_function, division

from os import environ, getcwd
from os import walk
from os.path import expanduser
from pdb import set_trace
import sys

# Update PYTHONPATH
HOME = expanduser('~')
axe = HOME + '/git/axe/axe/'  # AXE
pystat = HOME + '/git/pystats/'  # PySTAT
cwd = getcwd()  # Current Directory
sys.path.extend([axe, pystat, cwd, './old/Modeller/'])


import csv
from Prediction import *
from _imports import *
from abcd import _Abcd
from methods1 import *
import numpy as np
import pandas as pd
from pdb import set_trace

from Models._model import xomod, howMuchEffort
from Models._XOMO import *
from numpy import sum, array
from sk import rdivDemo
import Models.pom3
import random

from Planner.CROSSTREES import xtrees
from WHAT import treatments as HOW
from Planner.strawman import strawman


class XOMO():

  "XOMO"
  def __init__(i, n=0):
    i.header = Xomo(model="all").names
    i.depen = ['$>effort', '$>months', '$>defects', '$>risk']
    i.ndep = min(n, 3)

  def toCSV(i, data, dir='Models/Data/POM3/', name=None):
    with open(dir + name, 'w') as csvfile:
      writer = csv.writer(csvfile, delimiter=',',
                          quotechar='|', quoting=csv.QUOTE_MINIMAL)
      writer.writerow(
          ['$' + d for d in i.header] + [i.depen[i.ndep]])  # Header
      for cells in data[1]:  # Body
        indep = cells[:-4]
        depen = i.model(indep)
        body = indep + [depen]
        writer.writerow(body)
    return dir + name

  def genData(i, N=100):
    train = i.toCSV(xomod(N), name='Train.csv')
    test = i.toCSV(xomod(N), name='Test.csv')
    return [train], [test]

  def model(i, X):
    row = {h: el for h, el in zip(i.header, X)}
    return howMuchEffort(row, n=i.ndep)


class POM3():

  "POM3"
  def __init__(p3, n=0):
    p3.indep = ["Culture", "Criticality", "CriticalityModifier",
                "InitialKnown", "InterDependency", "Dynamism",
                "Size", "Plan", "TeamSize"]
    p3.depen = ['$>cost', '$<completion', '$>idle']
    p3.ndep = min(n, 2)

  def toCSV(p3, data, dir='Models/Data/POM3/', name=None):
    with open(dir + name, 'w') as csvfile:
      writer = csv.writer(csvfile, delimiter=',',
                          quotechar='|', quoting=csv.QUOTE_MINIMAL)
      writer.writerow(
          ['$' + d for d in p3.indep] + [p3.depen[p3.ndep]])  # Header
      for cells in data[1]:  # Body
        indep = cells[:-3]
        depen = p3.model(indep)
        body = indep + [depen]
        writer.writerow(body)
    return dir + name

  def genData(p3, N=100):
    train = p3.toCSV(pom3d(N), name='Train.csv')
    test = p3.toCSV(pom3d(N), name='Test.csv')
    return [train], [test]

  def model(p3, X):
    return pom3().simulate(X)[p3.ndep]


def predictor(tbl, n=0, Model=XOMO()):
  rows = [r.cells for r in tbl._rows]
  out = []
  for elem in rows:
    if Model.__doc__ == "XOMO":
      out += [Model(n=n).model(elem[:-2])]
    elif Model.__doc__ == "POM3":
      out += [Model(n=n).model(elem[:-2])]
  return out


def learner(mdl=XOMO, n=0, reps=24, numel=1000):
  train, test = mdl(n=0).genData(N=numel)
  for planner in ['dtree', 'HOW', 'baseln0', 'baseln1']:
    E = [planner]
    before = array(predictor(Model=mdl, n=n, tbl=createTbl(train)))
    after = lambda newTab: array(predictor(Model=mdl, n=n, tbl=newTab))
    frac = lambda aft: sum(aft) / sum(before)
    for _ in xrange(reps):
      "Apply Different Planners"
      if planner == 'xtrees':
        if mdl == POM3 and n == 1:
          newTab = xtrees(train=train,
                          test=test,
                          bin=False,
                          smoteit=False,
                          majority=True).main(which='Best')
        else:
          newTab = xtrees(train=train,
                          test=test,
                          bin=False,
                          smoteit=False,
                          majority=True).main(which='Best')
      if planner == 'dtree':
        if mdl == POM3 and n == 1:
          newTab = xtrees(train=train,
                          test=test,
                          bin=False,
                          smoteit=False,
                          majority=False).main(which='Best')
        else:
          newTab = xtrees(train=train,
                          test=test,
                          bin=False,
                          smoteit=False,
                          majority=False).main(which='Best')
      if planner == 'HOW':
        newTab = HOW(train=train, test=test).main()
      if planner == 'baseln0':
        newTab = strawman(
            train=train,
            test=test).main(mode='models')
      if planner == 'baseln1':
        newTab = strawman(
            train=train,
            test=test,
            prune=True).main(mode='models')
      E.append(frac(after(newTab)))
    yield E


def _test():
  for mdl in [POM3, XOMO]:
    print('## %s \n\n' % (mdl.__doc__))
    ndep = 4 if mdl == XOMO else 3
    random.seed(0)
    for n in xrange(ndep):
      print('#### %s \n```' % (mdl().depen[n][2:]))
      R = [r for r in learner(mdl, n, reps=28)]
      rdivDemo(R, isLatex=True)
      print('```')

if __name__ == "__main__":
  _test()
  #------- Debug --------
  set_trace()
