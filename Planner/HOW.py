#! /Users/rkrsn/miniconda/bin/python
from __future__ import print_function
from __future__ import division

from os import environ, getcwd
from os import walk
from os import remove as rm
from random import randint
from os.path import expanduser
from pdb import set_trace
import sys
from itertools import chain

# Update PYTHONPATH
HOME = expanduser('~')
axe = HOME + '/git/axe/axe/'  # AXE
pystat = HOME + '/git/pystats/'  # PySTAT
cwd = getcwd()  # Current Directory
sys.path.extend([axe, pystat, cwd])

from pdb import set_trace
from random import uniform as rand
from random import randint as randi
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from _imports.weights import weights as W
import csv

from Prediction import *
from smote import *
import _imports.makeAmodel as mam
from methods1 import *
import numpy as np
import pandas as pd
from counts import *
# import sk


class changes():

  def __init__(self):
    self.log = {}

  def save(self, name=None, old=None, new=None):
    if not old == new:
      self.log.update({name: (old, new)})

def genTable(tbl, rows):
  name = str(randint(0, 1e6))
  header = [h.name for h in tbl.headers[:-1]]
  with open('tmp0.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(header)
    for el in rows:
      writer.writerow(el[:-1])
  new = createTbl(['tmp0.csv'])
  rm('tmp0.csv')
  return new


class o:

  def __init__(i, **d):
    i.has().update(**d)

  def has(i):
    return i.__dict__

  def update(i, **d):
    i.has().update(d)
    return i

  def __repr__(i):
    show = [':%s %s' % (k, i.has()[k])
            for k in sorted(i.has().keys())
            if k[0] is not "_"]
    txt = ' '.join(show)
    if len(txt) > 60:
      show = map(lambda x: '\t' + x + '\n', show)
    return '{' + ' '.join(show) + '}'


def settings(**d):
  return o(
      name="WHAT",
      what="WHAT - A Contrast Set Planner",
      author="Rahul Krishna",
      copyleft="(c) 2014, MIT license, http://goo.gl/3UYBp",
      seed=1,
      f=None,
  ).update(**d)

opt = settings()


class vertex():

  def __init__(self, ID, rows):
    self._id = ID
    self.rows = rows
    self.represent = None

  def score(self):
    return np.mean([r.cells[-2] for r in self.rows])

  def representative(self):
    return [float(np.mean([k.cells[indx] for k in self.rows]))
            for indx in xrange(len(self.rows[0].cells) - 2)]


class treatments():

  def __init__(
          self,
          train,
          test,
          far=False,
          train_df=None,
          test_df=None,
          fSelect=True,
          Prune=False,
          infoPrune=0.5,
          extent=0.5):
    self.test, self.train = test, train
    self.extent = extent
    self.fSelect = fSelect
    self.Prune = Prune
    self.infoPrune = infoPrune
    self.far = far
    self.new_Tab = []
    self.train_df = train_df if train_df \
        else createTbl(self.train, isBin=False, bugThres=1)

    self.test_df = test_df if test_df \
        else createTbl(self.test, isBin=False, bugThres=1)

    self.change = []

  def clusterer(self):
    IDs = list(set([f.cells[-1] for f in self.train_df._rows]))
    clusters = []
    for _id in list(set(IDs)):

      clusters.append(vertex(ID=_id, rows=[f for f in self.train_df._rows
                                           if f.cells[-1] == _id]))
    return clusters

  def knn(self, one, two):
    pdistVect = []
#    set_trace()
    for ind, n in enumerate(two):
      pdistVect.append(
          [ind, euclidean(one.representative(), n.representative())])
    indices = sorted(pdistVect, key=lambda F: F[1], reverse=self.far)
    return [two[n[0]] for n in indices]

  def getHyperplanes(self):
    hyperPlanes = []
    ClusterRows = self.clusterer()
    while ClusterRows:
      one = ClusterRows.pop()
      try:
        two = self.knn(one, ClusterRows)[1]
      except IndexError:
        two = one
      hyperPlanes.append([one, two])
    return hyperPlanes

  def projection(self, node_one, node_two, three):
    if node_one.score() < node_two.score():
      one, two = node_one, node_two
    else:
      one, two = node_two, node_one
    plane = [b - a for a, b in zip(one.representative(), two.representative())]
    norm = np.linalg.norm(plane)
    unitVect = [p / norm for p in plane]
    proj = np.dot(three, unitVect)
    return proj

  def fWeight(self, criterion='Variance'):
    lbs = W(use=criterion).weights(self.train_df)
    sortedLbs = sorted([l / max(lbs[0]) for l in lbs[0]], reverse=True)
    indx = int(self.infoPrune * len(sortedLbs)) - 1 if self.Prune else -1
    cutoff = sortedLbs[indx]
    L = [l / max(lbs[0]) for l in lbs[0]]
    return [0 if l < cutoff else l for l in L] if self.Prune else L

  def mutate(self, me, others):
    C = changes()

    def one234(pop, f=lambda x: id(x)):
      seen = []

    if self.fSelect:
      for i, old, other, f in zip(range(len(me[:-2])), me[:-2], others.representative(), opt.f):
        C.save(
            self.train_df.headers[i].name[
                1:],
            old,
            new=old + self.extent * f * (other - old))

      self.change.append(C.log)
      # set_trace()
      return [my + self.extent * f * (good - my) for f, my, good in zip(opt.f, me[:-2], others.representative())]
    else:
      for i, old, other, f in zip(range(len(me[:-2])), me[:-2], others.representative(), opt.f):
        C.save(
            self.train_df.headers[i].name[
                1:],
            old,
            new=old + self.extent * (other - old))
      self.change.append(C.log)

      return [my + self.extent * (good - my)
              for f, my, good in zip(opt.f, me[:-2], others.representative())]

  def main(self, justDeltas=False):
    hyperPlanes = self.getHyperplanes()
    opt.f = self.fWeight()
    for rows in self.test_df._rows:
      newRow = rows
#       if rows.cells[-2] > 0:
      vertices = sorted(
          hyperPlanes,
          key=lambda F: self.projection(
              F[0],
              F[1],
              rows.cells[
                  :-2]),
          reverse=True)[0]
      [good, bad] = sorted(vertices, key=lambda F: F.score())
      newRow.cells[:-2] = self.mutate(rows.cells, good)
      self.new_Tab.append(newRow)

    if justDeltas:
      return self.change
    else:
      return genTable(self.test_df, rows=[r.cells for r in self.new_Tab])


def testPlanner2():
  dir = '../Data'
  one, two = explore(dir)
  fWeight = treatments(one[0], two[0]).fWeight(criterion='Variance')
  set_trace()

if __name__ == '__main__':
  testPlanner2()
