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
from WHAT import treatments
from _model import xomod
from _XOMO import *


class model:

  def __init__(i):
    pass

  def toCSV(i, data, dir='Data/', name=None):
    with open(dir + name, 'w') as csvfile:
      writer = csv.writer(csvfile, delimiter=',',
                          quotechar='|', quoting=csv.QUOTE_MINIMAL)
      writer.writerow(data[0])  # Header
      for cells in data[1]:  # Body
        writer.writerow(cells)
    return dir + name

  def genData(i):
    train = i.toCSV(xomod(N=100), name='Train.csv')
    test = i.toCSV(xomod(N=100), name='Test.csv')
    return createTbl([train]), createTbl([test])

  def XOMO(i, x):
    "XOMO"
    m = Model('xomoall')
    c = m.oo()
    scaleFactors = c.scaleFactors
    effortMultipliers = c.effortMultipliers
    defectRemovers = c.defectRemovers
    headers = scaleFactors + effortMultipliers + defectRemovers + ['kloc']
    bounds = {h: (c.all[h].min, c.all[h].max)
              for h in headers}
    a = c.x()['b']
    b = c.all['b'].y(a)

    def restructure(x):
      return {headers[i]: x[i] for i in xrange(len(headers))}

    def sumSfs(x, out=0, reset=False):
      for i in scaleFactors:
        out += x[i]
      return out

    def prodEms(x, out=1, reset=False):
      for i in effortMultipliers:
        out *= x[i]  # changed_nave
      return out

    def Sum(x):
      return sumSfs(restructure(x[1:-4]), reset=True)

    def prod(x):
      return c.prodEms(restructure(x[1:-4]), reset=True)

    def exp(x):
      return b + 0.01 * Sum(x)

    effort = lambda x: c.effort_calc(restructure(x[1:-4]),
                                     a=a, b=b, exp=exp(x),
                                     sum=Sum(x), prod=prod(x))
    months = lambda x: c.month_calc(restructure(x[1:-4]),
                                    effort(x), sum=Sum(x),
                                    prod=prod(x))
    defects = lambda x: c.defect_calc(restructure(x[1:-4]))
    risks = lambda x: c.risk_calc(restructure(x[1:-4]))

    return defects


if __name__ == "__main__":
  model().genData()
  #------- Debug --------
  set_trace()
