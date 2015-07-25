from __future__ import division
import sys
import os
from _curses import flash
sys.dont_write_bytecode = True
# Update PYTHONPATH
HOME = os.environ['HOME']
axe = HOME + '/git/axe/axe/'  # AXE
pystat = HOME + '/git/pystat/'  # PySTAT
cwd = os.getcwd()  # Current Directory
sys.path.extend([axe, pystat, cwd])
from lib    import *
from demos  import *
from counts import *
from table  import *
from fi     import *
import numpy as np

class prilims(object):

  def __init__(self):
    pass

  def unwrap(self, t0):
    """
    It creates a dictionary with table headers as keys and corresponding
    columns as the values. Example, T={'Header1': [1,2,3,4]; 'Header2':...}...
    you get the point.
    """
    rows = map(lambda x :x.cells, t0._rows)
    H = [];
    for i in t0.headers:
      H.append(i.__dict__['name'])
    rows.insert(0, H)
    return {rows[0][z]:[p[z] for p in rows[1:] if z <= len(rows[1]) - 1] \
             for z in xrange(0, len(rows[0]))}

  def chardiv(self, lst):
    """"
    This works by sorting all the variables in the alphabetical order and
    determining cuts based on index locations where the alphabets change.
    """
    def pairs(xs):
      for p in zip(xs[:-1], xs[1:]):
        yield p

    sortOrder = [i[0] for i in sorted(enumerate(lst[0])
                                      , key = lambda x:x[1]
                                      , reverse = False)]
    sortedIndep = [i[1] for i in sorted(enumerate(lst[0])
                                        , key = lambda x:x[1]
                                        , reverse = False)]
    sortedDep = [lst[1][z] for z in sortOrder]
    cuts = [];divs = []
    for x in xrange(1, len(sortedIndep)):
      if not sortedIndep[x - 1] == sortedIndep[x]:
        cuts.append(x)
    cuts.insert(0, 0); cuts.insert(len(cuts), len(sortOrder))
    for x, y in pairs(cuts):
      divs.append((sortedIndep[x], np.std(sortedDep[x:y])
                   , [(sortedIndep[z], sortedDep[z]) for z in xrange(x, y)]))
    return divs

  def sdiv(self, lst, tiny = 3, cohen = 0.3,
         num1 = lambda x:x[0], num2 = lambda x:x[1]):
    "Divide lst of (num1,num2) using variance of num2."
    #----------------------------------------------
    def divide(this, small):  # Find best divide of 'this'
      lhs, rhs = Num(), Num(num2(x) for x in this)
      n0, least, cut = 1.0 * rhs.n, rhs.sd(), None
      for j, x  in enumerate(this):
        if lhs.n > tiny and rhs.n > tiny:
          maybe = lhs.n / n0 * lhs.sd() + rhs.n / n0 * rhs.sd()
          if maybe < least :
            if abs(lhs.mu - rhs.mu) >= small:
              cut, least = j, maybe
        rhs - num2(x)
        lhs + num2(x)
      return cut, least
    #----------------------------------------------
    def recurse(this, small, cuts):
      cut, sd = divide(this, small)
      if cut:
        recurse(this[:cut], small, cuts)
        recurse(this[cut:], small, cuts)
      else:
        cuts += [(num1(this[0]), sd, this)]
      return cuts
    #---| main |-----------------------------------
    n = len(lst)
    small = Num(num2(x) for x in lst).sd() * cohen
    if lst:
      return recurse(sorted(lst, key = num1), small, [])


class weights(object):
  def __init__(self, use = 'Variance'):
    self.criterion = use
    self.p = prilims()
    pass
  def cuts(self, tbl):
    # Finds the cuts in each column of the table.
    T = self.p.unwrap(tbl)
    depen = [T[i.__dict__['name']] for i in [tbl.headers[-2]]]
    indep = [T[i.__dict__['name']] for i in tbl.headers[:-2]]
    flatten = lambda x: x if not isinstance(x, list) else x[0]  # If the list
    # contains elements such that each element in that list is a list of one
    # element like [[1,2,3]], flatten returns [1,2,3]
    def findcuts(lst1, lst2):
      #--- Determines whether to use sdiv or chardiv
      return self.p.sdiv(zip(lst1, lst2)) \
        if not isinstance(lst1[1], str) else self.p.chardiv([lst1, lst2])
    cuts1 = [findcuts(y, flatten(depen)) for y in indep]
    return cuts1, [i.__dict__['name'] for i in tbl.headers[:-2]]
  def weights(self, t0):
    # Compute weights based on minimum variance, note also that the weights are
    # normalized such that they add up to 1.
    a, names = self.cuts(t0)
    mVar = 0
    weights = []
    for k in a:
      for l in k:
        mVar += l[1]
      weights.append(mVar / len(k))
#     weights = (1 - weights / np.max(weights))
    return [abs(w - max(weights)) for w in weights], names

"""
class main:
  source='data/nasa93.csv'
  t0=table(source)
  ad=ahadist();
  a = ad.weights(t0)
  print a

if __name__=="__main__":
  main()
"""






