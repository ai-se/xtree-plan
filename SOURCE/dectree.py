from __future__ import division
import sys
import pdb
import os
from os import walk
sys.path.insert(0, os.getcwd() + '/_imports/')
import libWhere
sys.path.insert(1, '/User/rkrsn/git/axe/axe/')
from lib import *
sys.dont_write_bytecode = True
from dtree import *
from where2 import *
from table import *
from settingsWhere import *
from settings import *
from methods1 import *
from contrastset import *
import random
import numpy as np
import makeAmodel
randi = random.randint
rseed = random.seed
# from _imports import *

"""
def tdivPrec(where = None , dtree = None, train = None, test = None):
 rseed(1)
 makeaModel = makeAmodel.makeAModel()

 # pdb.set_trace()

"""
# Training
"""
 _r = []
 for t in train:
  m = makeaModel.csv2py(t)
  _r += m._rows
 m._rows = _r
 prepare(m, settings = where)  # Initialize all parameters for where2 to run
 tree = where2(m, m._rows)  # Decision tree using where2
 tbl = table(t)
 headerLabel = '=klass'
 Rows = []
 for k, _ in leaves(tree):  # for k, _ in leaves(tree):
  for j in k.val:
   tmp = (j.cells)
   tmp.append('_' + str(id(k) % 1000))
   j.__dict__.update({'cells': tmp})
   Rows.append(j.cells)
 tbl2 = newTable(tbl, headerLabel, Rows)


"""
# Testing
"""
 _r = []
 for tt in test:
  mTst = makeaModel.csv2py(tt)
  _r += mTst._rows
 mTst._rows = _r
 prepare(mTst, settings = where)  # Initialize all parameters for where2 to run
 tree = where2(mTst, mTst._rows)  # Decision tree using where2
 tbl = table(tt)
 headerLabel = '=klass'
 Rows = []
 for k, _ in leaves(tree):  # for k, _ in leaves(tree):
  for j in k.val:
   tmp = (j.cells)
   tmp.append('_' + str(id(k) % 1000))
   j.__dict__.update({'cells': tmp})
   Rows.append(j.cells)
 tbl3 = newTable(tbl, headerLabel, Rows)
 temp = []

 def sort(lst):
  return [i[0] for i in sorted(enumerate(lst), key = lambda x:x[1])], \
         [i[1] for i in sorted(enumerate(lst), key = lambda x:x[1])]

 def thresh(val1, val2):
  indx, sorted = sort()
 def isdefective(case, test = False):
  if not test:
   return 'Defect' if case.cells[-2] > 0 else 'No Defect'
  else:
   bugs = [r.cells[-2] for r in case.rows];
   meanBugs = np.mean(bugs);
   medianBugs = np.median(bugs);
   rangeBugs = (sorted(bugs)[0] + sorted(bugs)[-1]) / 2;
   temp.append(meanBugs);
   return 'Defect' if meanBugs > 1.5 else 'No Defect'

 testCase = tbl3._rows
 # print testCase

 testDefective = []
 defectivClust = []

 t = discreteNums(tbl2, map(lambda x: x.cells, tbl2._rows))
 myTree = tdiv(t, opt = dtree)
 # showTdiv(myTree)

 testCase = tbl3._rows
#   # print testCase

 for tC in testCase:
  loc = drop(tC, myTree)
  # if len(loc.kids)==0:
  testDefective.append(isdefective(tC))
  defectivClust.append(isdefective(loc, test = True))
 #
 # saveImg(temp, 10)

#   contrastSet = getContrastSet(loc, myTree)
#   print 'Contrast Set:', contrastSet
 return [testDefective, defectivClust]



def tdivPrec1(where = None , dtree = None, train = None, test = None):
 rseed(1)
 makeaModel = makeAModel()

 # pdb.set_trace()

"""
# Training
"""
 _r = []
 for t in train:
  m = makeaModel.csv2py(t)
  _r += m._rows
 # m._rows = _r
 # prepare(m, settings = where)  # Initialize all parameters for where2 to run
 # tree = where2(m, m._rows)  # Decision tree using where2
 tbl = table(t)
 headerLabel = '=klass'
 Rows = []
 for k in _r:  # for k, _ in leaves(tree):
   tmp = (k.cells)
   tmp.append('_' + str(tmp[-1]))
   k.__dict__.update({'cells': tmp})
   Rows.append(k.cells)
 tbl2 = newTable(tbl, headerLabel, Rows)


"""
# Testing
"""
 _r = []
 for tt in test:
  mTst = makeaModel.csv2py(tt)
  _r += mTst._rows
 # mTst._rows = _r
 # prepare(mTst, settings = where)  # Initialize all parameters for where2 to run
 # tree = where2(mTst, mTst._rows)  # Decision tree using where2
 tbl = table(tt)
 headerLabel = '=klass'
 Rows = []
 for k in _r:  # for k, _ in leaves(tree):
   tmp = (k.cells)
   tmp.append('_' + str(tmp[-1]))
   k.__dict__.update({'cells': tmp})
   Rows.append(k.cells)
 tbl3 = newTable(tbl, headerLabel, Rows)
 temp = []

 def sort(lst):
  return [i[0] for i in sorted(enumerate(lst), key = lambda x:x[1])], \
         [i[1] for i in sorted(enumerate(lst), key = lambda x:x[1])]

 def thresh(val1, val2):
  indx, sorted = sort()
 def isdefective(case, test = False):
  if not test:
   return 'Defect' if case.cells[-2] > 0 else 'No Defect'
  else:
   bugs = [r.cells[-2] for r in case.rows];
   meanBugs = np.mean(bugs);
   medianBugs = np.median(bugs);
   rangeBugs = (sorted(bugs)[0] + sorted(bugs)[-1]) / 2;
   temp.append(meanBugs);
   return 'Defect' if meanBugs > 1.5 else 'No Defect'

 testCase = tbl3._rows
 # print testCase

 testDefective = []
 defectivClust = []

 t = discreteNums(tbl2, map(lambda x: x.cells, tbl2._rows))
 myTree = tdiv(t, opt = dtree)
 # showTdiv(myTree)

 testCase = tbl3._rows
#   # print testCase

 for tC in testCase:
  loc = drop(tC, myTree)
  # if len(loc.kids)==0:
  testDefective.append(isdefective(tC))
  defectivClust.append(isdefective(loc, test = True))
 #
 # saveImg(temp, 10)

#   contrastSet = getContrastSet(loc, myTree)
#   print 'Contrast Set:', contrastSet
 return [testDefective, defectivClust]



if __name__ == '__main__':
 print 'dectree.py'

"""
