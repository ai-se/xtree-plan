#! /Users/rkrsn/miniconda/bin/python
from __future__ import print_function, division
from os import environ, getcwd
import sys

# Update PYTHONPATH
HOME = environ['HOME']
axe = HOME + '/git/axe/axe/'  # AXE
pystat = HOME + '/git/pystats/'  # PySTAT
cwd = getcwd()  # Current Directory
sys.path.extend([axe, pystat, cwd])

from Prediction import *
from Planning import *
from _imports import *
from abcd import _Abcd
from cliffsDelta import showoff
from contrastset import *
from dectree import *
from methods1 import *
import numpy as np
import pandas as pd
from sk import rdivDemo
from pdb import set_trace
from dEvol import tuner
from Planner2 import treatments as treatments2


def Bugs(tbl):
  cells = [i.cells[-2] for i in tbl._rows]
  return cells


def withinClass(data):
  N = len(data)
  return [(data[:n], [data[n]]) for n in range(1, N)]


def write(str):
  sys.stdout.write(str)


def printsk(res):
  "Now printing only g"
  tosh = []
  for p in res:
    dat1, dat2, dat3, dat4 = res[p]
    tosh.append([dat1[0][0]] + [k[-1] for k in dat1])
    tosh.append([dat2[0][0]] + [k[-1] for k in dat2])
    tosh.append([dat3[0][0]] + [k[-1] for k in dat3])
    tosh.append([dat4[0][0]] + [k[-1] for k in dat4])
  rdivDemo(tosh, isLatex=False)


def cliffsdelta(lst1, lst2):
  m, n = len(lst1), len(lst2)
  dom = lambda a, b: -1 if a < b else 1 if a > b else 0
  dominationMtx = [[dom(a, b) for a in lst1] for b in lst2]
  delta = sum([sum(b) for b in dominationMtx]) / (m * n)
  return delta


def main():
  dir = '../Data'
  from os import walk
  dataName = [Name for _, Name, __ in walk(dir)][0]
  numData = len(dataName)  # Number of data
  Prd = [CART]  # , rforest]  # , adaboost, logit, knn]
  _smoteit = [True]  # , False]
  _tuneit = [False]
  cd = {}
  abcd = []
  res = {}
  for n in xrange(numData):

    out11 = []
    outA1 = []
    out1 = []
    outFar = []
    outNear = []
    outa = []
    one, two = explore(dir)
    data = [one[i] + two[i] for i in xrange(len(one))]
    print('##', dataName[n])
    for p in Prd:
      train = [dat[0] for dat in withinClass(data[n])]
      test = [dat[1] for dat in withinClass(data[n])]
      reps = 10
      abcd = [[], []]
      for t in _tuneit:
        tunedParams = None if not t else params
        print('### Tuning') if t else print('### No Tuning')
        for _smote in _smoteit:
          #          for _n in xrange(0):
          _n = -1
          # Training data
          for _ in xrange(reps):

            train_DF = createTbl(train[_n], isBin=True)
#            set_trace()
            # Testing data
            test_df = createTbl(test[_n], isBin=True)
            predRows = []
            # Tune?
            actual = Bugs(test_df)
            before = p(train_DF, test_df,
                       tunings=tunedParams,
                       smoteit=True)
            tunedParams = None if not t else tuner(p, train[_n])
            for predicted, row in zip(before, test_df._rows):
              tmp = row.cells
              tmp[-2] = predicted
              if predicted > 0:
                predRows.append(tmp)
            predTest = clone(test_df, rows=predRows)
            # Find and apply contrast sets
#             newTab = treatments(train = train[_n],
#                                 test = test[_n],
#                                 verbose = False,
#                                 smoteit = False).main()

            newTab_near = treatments2(train=train[_n], far=False, test=test[_n]  # ).main()
                                      , test_df=predTest).main() \
                if predRows \
                else treatments2(train=train[_n], test=test[_n]).main()
            newTab_far = treatments2(train=train[_n], test=test[_n]  # ).main()
                                     , test_df=predTest).main() \
                if predRows \
                else treatments2(train=train[_n], test=test[_n]).main()

            after_far = p(train_DF, newTab_far,
                          tunings=tunedParams,
                          smoteit=True)
            after_near = p(train_DF, newTab_near,
                           tunings=tunedParams,
                           smoteit=True)
#             print(showoff(dataName[n], before, after))
            outa.append(_Abcd(before=actual, after=before))
#            set_trace()
            cliffsFar = cliffsdelta(Bugs(predTest), after_far)
            cliffsNear = cliffsdelta(Bugs(predTest), after_near)
#             print(cliffsDelta(Bugs(predTest), after))
#            print('Gain =  %1.2f' % float(\
#            	   (sum(Bugs(predTest)) - sum(after)) / sum(Bugs(predTest)) * 100), r'%')
            outFar.append(cliffsFar)
            outNear.append(cliffsNear)
#            out1.append(float((sum(before) - sum(after)) / sum(before) * 100))
#           out1 = [o for o in out1 if np.isfinite(o)]
          outNear.insert(0, dataName[n] + '_Far')
          outFar.insert(0, dataName[n] + '_Near')

          outa.insert(0, dataName[n])
        out11.extend([outNear, outFar])
        outA1.append(outa)
        try:
          print('```')
          rdivDemo(out11, isLatex=False)
    #      rdivDemo(outA1, isLatex = False)
          print('```')
        except IndexError:
          pass

if __name__ == '__main__':
  main()
