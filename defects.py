#! /Users/rkrsn/miniconda/bin/python
from __future__ import print_function, division

from os import environ, getcwd
from os import walk
from os.path import expanduser
from pdb import set_trace
import sys
from bdb import set_trace

# Update PYTHONPATH
HOME = expanduser('~')
axe = HOME + '/git/axe/axe/'  # AXE
pystat = HOME + '/git/pystats/'  # PySTAT
cwd = getcwd()  # Current Directory
sys.path.extend([axe, pystat, cwd])

import numpy as np
import pandas as pd
import csv
from random import seed as rseed
from abcd import _Abcd
#from cliffsDelta import cliffs
from _imports.dEvol import tuner
from demos import cmd
from sk import rdivDemo
from numpy import sum

from _imports import *
from Prediction import rforest, Bugs
from methods1 import *

from Planner.CROSSTREES import xtrees
from Planner.HOW import treatments as HOW
from Planner.strawman import strawman


def say(x):
  sys.stdout.write(str(x))


def write2file(data, fname='Untitled', ext='.txt'):
  with open('.temp/' + fname + ext, 'w') as fwrite:
    writer = csv.writer(fwrite, delimiter=',')
    if not isinstance(data[0], list):
      writer.writerow(data)
    else:
      for b in data:
        writer.writerow(b)


def genTable(tbl, rows, name='tmp'):
  header = [h.name for h in tbl.headers[:-1]]
  with open(name + '.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(header)
    for el in rows:
      if len(el[:-1]) < len(header):
        writer.writerow(el)
      else:
        writer.writerow(el[:-1])

  return createTbl([name + '.csv'])


class run():

  def __init__(
          self, pred=rforest, _smoteit=True, _n=-1, _tuneit=False, dataName=None, reps=1):
    self.pred = pred
    self.dataName = dataName
    self.out, self.out_pred = [self.dataName], []
    self._smoteit = _smoteit
    self.train, self.test = self.categorize()
    self.reps = reps
    self._n = _n
    self.tunedParams = None if not _tuneit \
        else tuner(self.pred, self.train[_n])
    self.headers = createTbl(
        self.train[
            self._n],
        isBin=False,
        bugThres=1).headers

  def categorize(self):
    dir = './Data/Jureczko'
    self.projects = [Name for _, Name, __ in walk(dir)][0]
    self.numData = len(self.projects)  # Number of data
    one, two = explore(dir)
    data = [one[i] + two[i] for i in xrange(len(one))]

    def withinClass(data):
      N = len(data)
      return [(data[:n], [data[n]]) for n in range(1, N)]

    def whereis():
      for indx, name in enumerate(self.projects):
        if name == self.dataName:
          return indx

    try:
      return [
          dat[0] for dat in withinClass(data[whereis()])], [
          dat[1] for dat in withinClass(data[whereis()])]  # Train, Test
    except:
      set_trace()

  def logResults(self, *args):
    for a in args:
      print(a)

  def go(self):
    base = lambda X: sorted(X)[-1] - sorted(X)[0]
    newRows = lambda newTab: map(lambda Rows: Rows.cells[:-1], newTab._rows)
    after = lambda newTab: self.pred(
        train_DF,
        newTab,
        tunings=self.tunedParams,
        smoteit=True)
    frac = lambda aft: 1 - (sum([0 if a < 1 else 1 for a in aft]) \
                            / sum([0 if b < 1 else 1 for b in actual]))

    for planner in ['XTREE', 'BIC', 'CD', 'CD+FS']:
      out = [planner]
      for _ in xrange(self.reps):
        predRows = []
        train_DF = createTbl(self.train[self._n], isBin=True)
        test_df = createTbl(self.test[self._n], isBin=True)
        actual = np.array(Bugs(test_df))
        before = self.pred(train_DF, test_df,
                           tunings=self.tunedParams,
                           smoteit=True)

        predRows = [row.cells for row in createTbl(
            self.test[self._n], isBin=True)._rows if row.cells[-2] > 0]

        predTest = genTable(test_df, rows=predRows, name='Before_temp')

        "Apply Different Planners"
        if planner == 'xtrees':
          newTab = xtrees(train=self.train[-1],
                          test_DF=predTest,
                          bin=False,
                          majority=True).main()
          genTable(test_df, rows=newRows(newTab), name='After_xtrees')
#          set_trace()
        elif planner == 'XTREE' or planner == 'XTREE':
          newTab = xtrees(train=self.train[-1],
                          test_DF=predTest,
                          bin=False,
                          majority=False).main()

        elif planner == 'BIC':
          newTab = HOW(train=self.train[-1],
                       test=self.test[-1],
                       test_df=predTest).main()

        elif planner == 'CD':
          newTab = strawman(train=self.train[-1], test=self.test[-1]).main()

        elif planner == 'CD+FS':
          newTab = strawman(train=self.train[-1], test=self.test[-1]
                            , prune=True).main()

        out.append(frac(after(newTab)))
#      self.logResults(out)
      yield out

# ---------- Debug ----------
#    set_trace()

  def delta1(self, cDict, headers, norm):
    for el in cDict:
      D = len(headers[:-2]) * [0]
      for k in el.keys():
        for i, n in enumerate(headers[:-1]):
          if n.name[1:] == k:
            D[i] += 100
      yield D

  def delta0(self, headers, norm, Planner='xtrees'):
    before, after = open('.temp/before.txt'), open('.temp/' + Planner + '.txt')
    D = len(headers[:-1]) * [0]
    for line1, line2 in zip(before, after):
      row1 = np.array([float(l) for l in line1.strip().split(',')[:-2]])
      row2 = np.array([float(l) for l in line2.strip().split(',')[:-1]])
      changed = (row2 - row1).tolist()
      for i, c in enumerate(changed):
        if c > 0:
          D[i] += 100
    return D

  def deltas(self, planner):
    delta = []
    train_DF = createTbl(self.train[self._n], isBin=True, bugThres=1)
    test_df = createTbl(self.test[self._n], isBin=True, bugThres=1)
    actual = np.array(Bugs(test_df))
    before = self.pred(train_DF, test_df, tunings=self.tunedParams,
                       smoteit=True)
    allRows = np.array(
        map(
            lambda Rows: np.array(
                Rows.cells[
                    :-
                    1]),
            train_DF._rows +
            test_df._rows))

    def min_max():
      N = len(allRows[0])
      base = lambda X: sorted(X)[-1] - sorted(X)[0]
      return [base([r[i] for r in allRows]) for i in xrange(N)]

    predRows = [row.cells for row in createTbl(
        self.test[self._n], isBin=True)._rows if row.cells[-2] > 0]

    write2file(predRows, fname='before')  # save file

    """
    Apply Learner
    """
    for _ in xrange(1):
      predTest = genTable(test_df, rows=predRows)

      newRows = lambda newTab: map(lambda Rows: Rows.cells[:-1], newTab._rows)

      "Apply Different Planners"
      if planner == 'xtrees':
        xTrees = xtrees(train=self.train[-1],
                        test_DF=predTest,
                        bin=False,
                        majority=True).main(justDeltas=True)
        delta.append(
            [d for d in self.delta1(xTrees, train_DF.headers, norm=len(predRows))])
        return (np.sum(
            delta[0], axis=0) / np.array((len(predRows[0]) - 2) * [len(predRows)])).tolist()

      elif planner == 'XTREE' or planner == 'XTREE':
        C4_5 = xtrees(train=self.train[-1],
                      test_DF=predTest,
                      bin=False, majority=False).main(justDeltas=True)

        delta.append(
            [d for d in self.delta1(C4_5, train_DF.headers, norm=len(predRows))])
        return (np.sum(
            delta[0], axis=0) / np.array((len(predRows[0]) - 2) * [len(predRows)])).tolist()

      elif planner == 'BIC':
        how = HOW(train=self.train[-1],
                  test=self.test[-1],
                  test_df=predTest).main(justDeltas=True)
        delta.append(
            [d for d in self.delta1(how, train_DF.headers, norm=len(predRows))])
        return (np.sum(
            delta[0], axis=0) / np.array((len(predRows[0]) - 2) * [len(predRows)])).tolist()

      elif planner == 'CD':
        baseln = strawman(
            train=self.train[-1], test=self.test[-1]).main(justDeltas=True)
        delta.append(
            [d for d in self.delta1(baseln, train_DF.headers, norm=len(predRows))])
        return (np.sum(
            delta[0], axis=0) / np.array((len(predRows[0]) - 2) * [len(predRows)])).tolist()

      elif planner == 'CD+FS':
        baselnFss = strawman(
            train=self.train[-1], test=self.test[-1], prune=True).main(justDeltas=True)
        delta.append(
            [d for d in self.delta1(baselnFss, train_DF.headers, norm=len(predRows))])
        return (np.sum(
            delta[0], axis=0) / np.array((len(predRows[0]) - 2) * [len(predRows)])).tolist()

   # -------- DEBUG! --------
    # set_trace()


def deltaCSVwriter0():
  Planners = ['XTREE', 'BIC', 'CD', 'CD+FS']
  print(',%s,%s,%s,%s' % tuple(Planners))
  for name in ['ant', 'ivy', 'jedit', 'lucene', 'poi']:
    say(name)
    delta = []
    R = run(dataName=name, reps=1)  # Setup Files.
    for p in Planners:
      delta.append(R.deltas(planner=p))

    D = np.mean(delta, axis=1).tolist()
    for n in D:
      say(',%0.2f' % (n))
    print('')
#       set_trace()


def deltaCSVwriter(type='Indv'):

  if type == 'Indv':
    for name in ['lucene']:
      print('##', name)
      delta = []
      Planners = ['XTREE', 'BIC', 'CD', 'CD+FS']
      R = run(dataName=name, reps=1)  # Setup Files.

      for p in Planners:
        delta.append(R.deltas(planner=p))

      def getRow(i):
        for d in delta:
          yield d[i]

#       set_trace()
      with open('/Users/rkrsn/git/GNU-Plots/rkrsn/errorbar/%s.csv' %
                (name), 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ')
        writer.writerow(["Features"] + Planners)
        for i, h in enumerate(run(dataName=name).headers[:-2]):
          writer.writerow([h.name[1:]] + [el for el in getRow(i)])
#      set_trace()
  elif type == 'All':
    delta = []
    for name in ['ivy', 'jedit', 'lucene', 'poi', 'ant']:
      print('##', name)
      delta.extend(run(dataName=name, reps=4).deltas())
    y = np.median(delta, axis=0)
    yhi, ylo = np.percentile(delta, q=[75, 25], axis=0)
    dat1 = sorted([(h.name[1:], a, b, c) for h, a, b, c in zip(
        run(dataName=name).headers[:-2], y, ylo, yhi)], key=lambda F: F[1])
    dat = np.asarray([(d[0], n, d[1], d[2], d[3])
                      for d, n in zip(dat1, range(1, 21))])
    with open('/Users/rkrsn/git/GNU-Plots/rkrsn/errorbar/all.csv', 'w') as csvfile:
      writer = csv.writer(csvfile, delimiter=' ')
      for el in dat[()]:
        writer.writerow(el)


def rdiv():
  lst = []

  def striplines(line):
    listedline = line.strip().split(',')  # split around the = sign
    listedline[0] = listedline[0][2:-1]
    lists = [listedline[0]]
    for ll in listedline[1:-1]:
      lists.append(float(ll))
    return lists

  f = open('./jedit.dat')
  for line in f:
    lst.append(striplines(line[:-1]))

  rdivDemo(lst, isLatex=False)
  set_trace()


def deltaTest():
  for file in ['ivy', 'poi', 'jedit', 'ant', 'lucene']:
    print('##', file)
    R = run(dataName=file, reps=12).deltas()

def _test(file='ant'):
  rseed(1)
  for file in ['ivy', 'lucene', 'jedit', 'poi', 'ant']:
    print('## %s\n' % (file))
    R = [r for r in run(dataName=file, reps=40).go()]
    rdivDemo(R, isLatex=False)
    
if __name__ == '__main__':
    _test()
  # deltaTest()
  # rdiv()
  # deltaCSVwriter(type='All')
  #   deltaCSVwriter(type='Indv')
#  deltaCSVwriter0()
#   eval(cmd())
