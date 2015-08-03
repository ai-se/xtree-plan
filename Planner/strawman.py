#! /Users/rkrsn/miniconda/bin/python
from __future__ import print_function, division
from numpy import array, asarray, mean, median, percentile, size, sum, sqrt
from pdb import set_trace
from methods1 import createTbl
from Prediction import rforest, rforest2
from _imports.weights import weights as W
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
sys.path.extend([axe, pystat, cwd])

from table import clone
import csv


def eDist(row1, row2):
  "Euclidean Distance"
  return sum([(a * a - b * b) ** 0.5 for a, b in zip(row1[:-1], row2[:-1])])


class node():

  """
  A data structure to hold all the rows in a cluster.
  Also return an exemplar: the centroid.
  """

  def __init__(self, rows):
    self.rows = []
    for r in rows:
      self.rows.append(r.cells[:-1])

  def exemplar(self, what='centroid'):
    if what == 'centroid':
      return median(array(self.rows), axis=0)
    elif what == 'mean':
      return mean(array(self.rows), axis=0)


class contrast():

  "Identify the nearest enviable node."

  def __init__(self, clusters):
    self.clusters = clusters

  def closest(self, testCase):
    return sorted([f for f in self.clusters],
                  key=lambda F: eDist(F.exemplar(), testCase.cells[:-1]))[0]

  def envy(self, testCase, alpha=0.5):
    me = self.closest(testCase)
    others = [o for o in self.clusters if not me == o]
    betters = [f for f in others if f.exemplar()[-1] <= me.exemplar()[-1]]
    try:
      return sorted([f for f in betters],
                    key=lambda F: eDist(F.exemplar(), me.exemplar()))[0]
    except:
      return me


class patches():

  "Apply new patch."

  def __init__(
          self, train, test, clusters, prune=False, B=0.25, verbose=False, bin=False, pred=[]):
    if bin:
      self.train = createTbl(train, isBin=False)
      self.test = createTbl(test, isBin=False)
    else:
      self.train = createTbl(train, isBin=True)
      self.test = createTbl(test, isBin=True)

    self.clusters = clusters
    self.Prune = prune
    self.B = B
    self.mask = self.fWeight()
    self.write = verbose
    self.bin = bin
    self.pred = pred

  def min_max(self):
    allRows = array(
        map(
            lambda Rows: array(
                Rows.cells[
                    :-
                    2]),
            self.train._rows +
            self.test._rows))
    N = len(allRows[0])
    base = lambda X: sorted(X)[-1] - sorted(X)[0]
    return array([base([r[i] for r in allRows]) for i in xrange(N)])

  def fWeight(self, criterion='Variance'):
    lbs = W(use=criterion).weights(self.train)
    try:
      sortedLbs = sorted([l / max(0.0001, max(lbs[0]))
                          for l in lbs[0]], reverse=True)
    except:
      set_trace()
    indx = int(self.B * len(sortedLbs)) - 1 if self.Prune else -1
    cutoff = sortedLbs[indx]
    L = [l / max(0.0001, max(lbs[0])) for l in lbs[0]]
    return array([0 if l < cutoff else l for l in L] if self.Prune else L)

  def delta0(self, node1, node2):
    if not self.bin:
      return array([el1 - el2 for el1, el2 in zip(node1.exemplar()
                                                  [:-1], node2.exemplar()[:-1])]) / self.min_max() * self.mask
    else:
      return array([el1 == el2 for el1, el2 in zip(node1.exemplar()
                                                   [:-1], node2.exemplar()[:-1])])

  def delta(self, t):
    C = contrast(self.clusters)
    closest = C.closest(t)
    better = C.envy(t, alpha=0.5)
    return self.delta0(closest, better)

  def patchIt(self, t):
    if not self.bin:
      return (array(t.cells[:-2]) + self.delta(t)).tolist()
    else:
      return [1 - val if d and m > 0 else val for val, m,
              d in zip(t.cells[:-2], self.mask, self.delta(t))]

  def newTable(self):
    if not self.bin:
      oldRows = [r for r in self.test._rows if r.cells[-2] > 0]
    else:
      oldRows = self.test._rows
    newRows = [self.patchIt(t) for t in oldRows]

    if self.write:
      self.deltasCSVWriter()

    header = [h.name for h in self.test.headers[:-1]]
    with open('tmp0.csv', 'w') as csvfile:
      writer = csv.writer(csvfile, delimiter=',')
      writer.writerow(header)
      for el in newRows:
        writer.writerow(el + [0])

    return createTbl(['tmp0.csv'])

  def deltasCSVWriter(self, name='ant'):
    "Changes"
    header = array([h.name[1:] for h in self.test.headers[:-2]])
    oldRows = [r for r, p in zip(self.test._rows, self.pred) if p > 0]
    delta = array([self.delta(t) for t in oldRows])
    y = median(delta, axis=0)
    yhi, ylo = percentile(delta, q=[75, 25], axis=0)
    dat1 = sorted(
        [(h, a, b, c) for h, a, b, c in zip(header, y, ylo, yhi)], key=lambda F: F[1])
    dat = asarray([(d[0], n, d[1], d[2], d[3])
                   for d, n in zip(dat1, range(1, 21))])
    with open('/Users/rkrsn/git/GNU-Plots/rkrsn/errorbar/%s.csv' % (name), 'w') as csvfile:
      writer = csv.writer(csvfile, delimiter=' ')
      for el in dat[()]:
        writer.writerow(el)
    # new = [self.newRow(t) for t in oldRows]


class strawman():

  def __init__(self, train, test, prune=False):
    self.train, self.test = train, test
    self.prune = prune

  def nodes(self, rowObject):
    clusters = set([r.cells[-1] for r in rowObject])
    for id in clusters:
      cluster = []
      for row in rowObject:
        if row.cells[-1] == id:
          cluster.append(row)
      yield node(cluster)

  def main(self, config=False):
    if not config:
      train_DF = createTbl(self.train, isBin=True)
      test_DF = createTbl(self.test, isBin=True)
      before = rforest(train=train_DF, test=test_DF)
#      set_trace()
      clstr = [c for c in self.nodes(train_DF._rows)]
      return patches(train=self.train,
                     test=self.test,
                     clusters=clstr,
                     prune=self.prune,
                     pred=before).newTable()
    else:
      train_DF = createTbl(self.train, isBin=False)
      test_DF = createTbl(self.test, isBin=False)
      before = rforest2(train=train_DF, test=test_DF)
      clstr = [c for c in self.nodes(train_DF._rows)]
      return patches(train=self.train,
                     test=self.test,
                     clusters=clstr,
                     prune=self.prune,
                     pred=before,
                     bin=True).newTable()


def categorize(dataName):
  dir = '../Data/Jureczko'
  projects = [Name for _, Name, __ in walk(dir)][0]
  numData = len(projects)  # Number of data
  one, two = explore(dir)
  data = [one[i] + two[i] for i in xrange(len(one))]

  def withinClass(data):
    N = len(data)
    return [(data[:n], [data[n]]) for n in range(1, N)]

  def whereis():
    for indx, name in enumerate(projects):
      if name == dataName:
        return indx

  try:
    return [
        dat[0] for dat in withinClass(data[whereis()])], [
        dat[1] for dat in withinClass(data[whereis()])]  # Train, Test
  except:
    set_trace()

if __name__ == '__main__':
  for name in ['ivy', 'jedit', 'lucene', 'poi', 'ant']:
    train, test = categorize(name)
    strawman(train[-1], test[-1]).main()
