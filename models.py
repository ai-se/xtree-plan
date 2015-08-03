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
from Models.pom3 import pom3
import random

from Planner.xtress_bin import xtrees
from WHAT import treatments as HOW
from Planner.strawman import strawman


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
    return train, test

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
    return train, test

  def model(p3, X):
    try:
      return pom3().simulate(X)[0]
    except:
      set_trace()


def predictor(tbl, Model=XOMO):
  rows = [r.cells for r in tbl._rows]
  out = []
  for elem in rows:
    if Model == XOMO:
      out += [Model().model(elem[:-2])]
    elif Model == POM3:
      out += [Model().simulate(elem[:-2])[0]]
  return out


def learner(mdl=XOMO, lst=[], reps=24):
  train, test = mdl().genData(N=1000)
  before = array(predictor(Model=mdl, tbl=createTbl(train)))
  for planner in ['xtrees', 'cart', 'HOW', 'baseln0', 'baseln1']:
    E = [planner]
    after = lambda newTab: array(predictor(tbl=newTab))
    frac = lambda aft: sum(aft) / sum(before)
    for _ in xrange(reps):
      "Apply Different Planners"
      if planner == 'xtrees':
        newTab = xtrees(train=train,
                        test=test,
                        bin=False,
                        majority=True).main()
      if planner == 'cart':
        newTab = xtrees(train=train,
                        test=test,
                        bin=False,
                        majority=False).main()
      if planner == 'HOW':
        newTab = HOW(train=train, test=test).main()
      if planner == 'baseln0':
        newTab = strawman(
            train=train,
            test=test).main(config=True)
      if planner == 'baseln1':
        newTab = strawman(
            train=train,
            test=test,
            prune=True).main(config=True)
      E.append(sum(after) / sum(before))
      lst.append(E)
  return lst

if __name__ == "__main__":
  random.seed(0)
  lst = learner(reps=10)
  try:
    rdivDemo(lst, isLatex=True)
  except:
    set_trace()
  #------- Debug --------
  set_trace()
