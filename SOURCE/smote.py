#! /Users/rkrsn/anaconda/bin/python
from pdb import set_trace
from os import environ, getcwd
import sys
from scipy.spatial.distance import euclidean
# Update PYTHONPATH
HOME = environ['HOME']
axe = HOME + '/git/axe/axe/'  # AXE
pystat = HOME + '/git/pystats/'  # PySTAT
cwd = getcwd()  # Current Directory
sys.path.extend([axe, pystat, cwd])
from random import choice, seed as rseed, uniform as rand
import pandas as pd


def SMOTE(data=None, k=5, atleast=10, atmost=51, bugIndx=2, resample=False):

  def Bugs(tbl):
    cells = [i.cells[-bugIndx] for i in tbl._rows]
    return cells

  def minority(data):
    unique = list(set(sorted(Bugs(data))))
    counts = len(unique) * [0]
#     set_trace()
    for n in xrange(len(unique)):
      for d in Bugs(data):
        if unique[n] == d:
          counts[n] += 1
    return unique, counts

  def knn(one, two):
    pdistVect = []
#    set_trace()
    for ind, n in enumerate(two):
      pdistVect.append([ind, euclidean(one.cells[:-1], n.cells[:-1])])
    indices = sorted(pdistVect, key=lambda F: F[1])
    return [two[n[0]] for n in indices]

  def extrapolate(one, two):
    new = one
#    set_trace()
    if bugIndx == 2:
      new.cells[3:-1] = [max(min(a, b),
                             min(min(a, b) + rand() * (abs(a - b)),
                                 max(a, b))) for a, b in zip(one.cells[3:-1],
                                                             two.cells[3:-1])]
      new.cells[-2] = int(new.cells[-2])
    else:
      new.cells[3:] = [min(a, b) + rand() * (abs(a - b)) for
                       a, b in zip(one.cells[3:], two.cells[3:])]
      new.cells[-1] = int(new.cells[-1])
    return new

  def populate(data):
    newData = []
    # reps = (len(data) - atleast)
    for _ in xrange(atleast):
      for one in data:
        neigh = knn(one, data)[1:k + 1]
        # If you're thinking the following try/catch statement is bad coding
        # etiquette i i .
        try:
          two = choice(neigh)
        except IndexError:
          two = one
        newData.append(extrapolate(one, two))
    # data.extend(newData)
    return newData

  def depopulate(data):
    if resample:
      newer = []
      for _ in xrange(atmost):
        orig = choice(data)
        newer.append(extrapolate(orig, knn(orig, data)[1]))
      return newer
    else:
      return [choice(data) for _ in xrange(atmost)]

  newCells = []
  rseed(1)
  unique, counts = minority(data)
  rows = data._rows
  for u, n in zip(unique, counts):
    if n < atleast:
      newCells.extend(populate([r for r in rows if r.cells[-2] == u]))
    if n > atmost:
      newCells.extend(depopulate([r for r in rows if r.cells[-2] == u]))
    else:
      newCells.extend([r for r in rows if r.cells[-2] == u])

  return clone(data, rows=[k.cells for k in newCells])


def test_smote():
  dir = '../Data/camel/camel-1.6.csv'
  Tbl = createTbl([dir], _smote=False)
  newTbl = createTbl([dir], _smote=True)
  print(len(Tbl._rows), len(newTbl._rows))
  # for r in newTbl._rows:
  #   print r.cells

if __name__ == '__main__':
  test_smote()
