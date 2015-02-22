#! /Users/rkrsn/anaconda/bin/python
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
from deTuner import tuner

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
  rdivDemo(tosh, isLatex = False)

def main():
  dir = '../Data'
  from os import walk
  dataName = [Name for _, Name, __ in walk(dir)][0]
  numData = len(dataName)  # Number of data
  Prd = [CART]  # , rforest]  # , adaboost, logit, knn]
  _smoteit = [True]  # , False]
  _tuneit = [False]  # , False]
  cd = {}
  abcd = []
  res = {}
  out11 = [];
  outA1 = [];
  for n in xrange(numData):
    out1 = [];
    outa = []
    one, two = explore(dir)
    data = [one[i] + two[i] for i in xrange(len(one))];
    print('##', dataName[n])
    for p in Prd:
#       print(p.__doc__)
      # params = tuner(p, data[0])
#       print(params)
      train = [dat[0] for dat in withinClass(data[n])]
      test = [dat[1] for dat in withinClass(data[n])]
      reps = 4
      abcd = [[], []];
      for t in _tuneit:
        tunedParams = None
        print('### Tuning') if t else print('### No Tuning')
        for _smote in _smoteit:
#           print('### SMOTE-ing') if _smote else print('### No SMOTE-ing')
    #       print('```')
#          for _n in xrange(0):
#          set_trace()
          _n = -1
          # Training data
          for _ in xrange(reps):

            train_DF = createTbl(train[_n], isBin = True)
#            set_trace()
            # Testing data
            test_df = createTbl(test[_n], isBin = True)
            # Tune?
#             tunedParams = None if not t else params
            # Find and apply contrast sets
            newTab = treatments(train = train[_n],
                                test = test[_n],
                                verbose = False,
                                smoteit = True).main()
#

            # set_trace()
            # Actual bugs
            actual = Bugs(test_df)
            # actual1 = [0 if a < 2 else 1 for a in actual]
            # Use the classifier to predict the number of bugs in the raw data.
            before = p(train_DF, test_df,
                       tunings = tunedParams,
                       smoteit = True)
            # before1 = [0 if b < 1 else 1 for b in before]
            # Use the classifier to predict the number of bugs in the new data.
            after = p(train_DF, newTab,
                      tunings = tunedParams,
                      smoteit = _smote)
            # after1 = [0 if a < 2 else 1 for a in after]

            # set_trace()

#             write('.')
#             write('Training: '); [write(l + ', ') for l in train[_n]]; print('\n')
#             cd.append(showoff(dataName[n], actual1, after1))
#             print(showoff(dataName[n], before, after))
#             write('Test: '); [write(l) for l in test[_n]],
            outa.append(_Abcd(before = actual, after = before)[-1])
          #   print(outa)
          #   print('Gain =  %0.2f' % float(\
          #   	(sum(before) - sum(after)) / sum(before)*100),r'%')

          #   out1.append(\
          #   	float((sum(before) - sum(after)) / sum(before)*100))

          # out1 = [o for o in out1 if np.isfinite(o)]
          # out1.insert(0, dataName[n])

          outa.insert(0, dataName[n])
#             %print('Prediction accuracy (g)  %.2d' % out[-1])
#             print (out[-1])
    #         if _smote:
    #           out.insert(0, p.__doc__ + '(s, Tuned)  ') if t \
    #           else out.insert(0, p.__doc__ + '(s, Naive)  ')
    #           abcd[0].append(out)
    #         else:
    #           out.insert(0, p.__doc__ + '(raw, Tuned)') if t \
    #           else out.insert(0, p.__doc__ + '(raw, Naive)')
    #           abcd[1].append(out)
    # print(out1)
    # out11.append(out1)
    outA1.append(outa)
      # print()
      # cd.update({p.__doc__:sorted(cd)})
      # res.update({p.__doc__:(abcd[0][0:reps],
      #                      abcd[0][reps:] ,
      #                      abcd[1][0:reps],
      #                      abcd[1][reps:])})
  # rdivDemo(out11, isLatex = True)
    # print('```')
    # rdivDemo(outA1, isLatex = False)
    # print('```')

    print('```')
    rdivDemo(outA1, isLatex = False)
# #     print(cd)
    # printsk(res)
    print('```')

          # sk.rdivDemo(stat)
          # Save the histogram after applying contrast sets.
          # saveImg(bugs, num_bins = 10, fname = 'bugsAfter', ext = '.jpg')

          # <<DEGUG: Halt Code>>
     # cd.update({p.__doc__:sorted(cd)})
      # set_trace()

if __name__ == '__main__':
#  dir = '../Data'
#  one, two = explore(dir)
#  data = [one[i] + two[i] for i in xrange(len(one))];
#  for p in [CART, rforest]:
#    params = tuner(p, data[0])
#    print (params)
# #  _CART()k
# #  _logit()
# #  _adaboost()
  main()
