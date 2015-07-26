#! /Users/rkrsn/miniconda/bin/python
from __future__ import print_function, division

from os import environ, getcwd
from os import walk
from pdb import set_trace
import sys

# Update PYTHONPATH
HOME = environ['HOME']
axe = HOME + '/git/axe/axe/'  # AXE
pystat = HOME + '/git/pystats/'  # PySTAT
cwd = getcwd()  # Current Directory
sys.path.extend([axe, pystat, cwd])

import numpy as np
import pandas as pd
import csv

from abcd import _Abcd
from cliffsDelta import cliffs
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


def write2file(data, fname='Untitled', ext='.txt'):
  with open(fname + ext, 'w') as fwrite:
    writer = csv.writer(fwrite, delimiter=',')
    if not isinstance(data[0], list):
      writer.writerow(data)
    else:
      for b in data:
        writer.writerow(b)


def genTable(tbl, rows):
  header = [h.name for h in tbl.headers[:-1]]
  with open('tmp.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(header)
    for el in rows:
      writer.writerow(el[:-1])

  return createTbl(['tmp.csv'])


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

  def go(self):
    out_xtrees = ['xtrees']
    out_HOW = ['HOW']
    out_basln = ['Base']
    out_baslnFss = ['Base+FSS']
    for _ in xrange(self.reps):
      predRows = []
      predRows1 = []
      train_DF = createTbl(self.train[self._n], isBin=True)
      test_df = createTbl(self.test[self._n], isBin=True)
      actual = np.array(Bugs(test_df))
      before = self.pred(train_DF, test_df,
                         tunings=self.tunedParams,
                         smoteit=True)

      for predicted, row in zip(before, test_df._rows):
        tmp = row.cells
        tmp[-2] = predicted
        if predicted > 0:
          predRows.append(tmp)

      predRows1 = [row.cells for predicted,
                   row in zip(before, test_df._rows) if predicted > 0]

#       set_trace()
      predTest = genTable(test_df, rows=predRows1)

      "Apply Different Planners"
      xTrees = xtrees(train=self.train[-1], test_DF=predTest, bin=False).main()
      how = HOW(train=self.train[-1],
                test=self.test[-1],
                test_df=predTest).main()
      baseln = strawman(train=self.train[-1], test=self.test[-1]).main()
      baselnFss = strawman(
          train=self.train[-1], test=self.test[-1], prune=True).main()

      after = lambda newTab: self.pred(train_DF, newTab,
                                       tunings=self.tunedParams,
                                       smoteit=True)
#       frac = lambda aft: sum([0 if a < 1 else 1 for a in aft]) / \
#           sum([0 if b < 1 else 1 for b in before])
      frac = lambda aft: sum(aft) / sum(before)

#       set_trace()
      out_xtrees.append(frac(after(xTrees)))
      out_HOW.append(frac(after(how)))
      out_basln.append(frac(after(baseln)))
      out_baslnFss.append(frac(after(baselnFss)))
    print(out_xtrees, '\n', out_HOW, '\n', out_basln, '\n', out_baslnFss)
    # ---------- Debug ----------

  def delta0(self, norm):
    before, after = open('before.txt'), open('after.txt')
    for line1, line2 in zip(before, after):
      row1 = np.array([float(l) for l in line1.strip().split(',')[:-1]])
      row2 = np.array([float(l) for l in line2.strip().split(',')])
      yield ((row2 - row1) / norm).tolist()

  def deltas(self):
    predRows = []
    delta = []
    train_DF = createTbl(self.train[self._n], isBin=True, bugThres=1)
    test_df = createTbl(self.test[self._n], isBin=True, bugThres=1)
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

    for predicted, row in zip(before, test_df._rows):
      tmp = row.cells
      tmp[-2] = predicted
      if predicted > 0:
        predRows.append(tmp)

    write2file(predRows, fname='before')  # save file

    """
    Apply Learner
    """
    for _ in xrange(1):
      predTest = clone(test_df, rows=predRows)
      newTab = xtrees(train=self.train[self._n], test_DF=predTest).main()
      newRows = np.array(map(lambda Rows: Rows.cells[:-1], newTab._rows))
      write2file(newRows, fname='after')  # save file
      delta.append([d for d in self.delta0(norm=min_max())])

    return delta[0]

    # -------- DEBUG! --------
    # set_trace()


def _test(file='ant'):
  for file in ['ivy', 'jedit', 'lucene', 'poi', 'ant']:
    print('##', file)
    R = run(dataName=file, reps=5).go()


def deltaCSVwriter(type='Indv'):
  if type == 'Indv':
    for name in ['ivy', 'jedit', 'lucene', 'poi', 'ant']:
      print('##', name)
      delta = run(dataName=name, reps=4).deltas()
      y = np.median(delta, axis=0)
      yhi, ylo = np.percentile(delta, q=[75, 25], axis=0)
      dat1 = sorted([(h.name[1:], a, b, c) for h, a, b, c in zip(
          run(dataName=name).headers[:-2], y, ylo, yhi)], key=lambda F: F[1])
      dat = np.asarray([(d[0], n, d[1], d[2], d[3])
                        for d, n in zip(dat1, range(1, 21))])
      with open('/Users/rkrsn/git/GNU-Plots/rkrsn/errorbar/%s.csv' % (name), 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ')
        for el in dat[()]:
          writer.writerow(el)
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
    R = run(dataName=file, reps=10).deltas()


if __name__ == '__main__':
  _test()
# deltaTest()
# rdiv()
# deltaCSVwriter(type='All')
# deltaCSVwriter(type='Indv')
#  eval(cmd())
