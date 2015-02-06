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
from random import choice, seed, uniform as rand
import pandas as pd
from dectree import *


def SMOTE(data = None, k = 5, atleast = 50, atmost = 100):

  def Bugs(tbl):
    cells = [i.cells[-2] for i in tbl._rows]
    return cells

  def minority(data):
    unique = list(set(sorted(Bugs(data))))
    counts = len(unique) * [0];
#     set_trace()
    for n in xrange(len(unique)):
      for d in Bugs(data):
        if unique[n] == d: counts[n] += 1
    return unique, counts

  def knn(one, two):
    pdistVect = []
#    set_trace()
    for ind, n in enumerate(two):
      pdistVect.append([ind, euclidean(one.cells[:-1], n.cells[:-1])])
    indices = sorted(pdistVect, key = lambda F:F[1])
    return [two[n[0]] for n in indices]

  def extrapolate(one, two):
    new = one;
#    set_trace()
    new.cells[3:-1] = [min(a, b) + rand() * (abs(a - b)) for
           a, b in zip(one.cells[3:-1], two.cells[3:-1])]
    new.cells[-2] = int(new.cells[-2])
    return new

  def populate(data):
    newData = []
    reps = len(data) - atleast
    for _ in xrange(reps):
      for one in data:
        neigh = knn(one, data)[1:k + 1];
        two = choice(neigh)
        newData.append(extrapolate(one, two))
    data.extend(newData)
    return data

  def depopulate(data):
    return [choice(data) for _ in xrange(atmost)]

  newCells = []
  seed(1)
  unique, counts = minority(data)
  rows = data._rows
  for u, n in zip(unique, counts):
    if  1 < n < atleast:
      newCells.extend(populate([r for r in rows if r.cells[-2] == u]))
    elif n > atmost:
      newCells.extend(depopulate([r for r in rows if r.cells[-2] == u]))
    else:
      newCells.extend([r for r in rows if r.cells[-2] == u])

  return clone(data, rows = [k.cells for k in newCells])

def test_smote():
  dir = '../Data/camel/camel-1.6.csv'
  Tbl = createTbl([dir])
  newTbl = SMOTE(data = Tbl)
  for r in newTbl._rows:
    print r.cells

if __name__ == '__main__':
  test_smote()
