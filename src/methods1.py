from dtree import *
from table import *

from _imports.where2 import *
from makeAmodel import makeAModel
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


def newTable(tbl, headerLabel, Rows):
 tbl2 = clone(tbl)
 newHead = Sym()
 newHead.col = len(tbl.headers)
 newHead.name = headerLabel
 tbl2.headers = tbl.headers + [newHead]
 return clone(tbl2, rows = Rows)

def createTbl(data):
 makeaModel = makeAModel()
 _r = []
 for t in data:
  m = makeaModel.csv2py(t)
  _r += m._rows
 m._rows = _r
 prepare(m, settings = None)  # Initialize all parameters for where2 to run
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
 return newTable(tbl, headerLabel, Rows)

def drop(test, tree):
 loc = apex(test, tree)
 return loc

def saveImg(x, num_bins = 10, fname = None, ext = None):
  pass
#  fname = 'Untitled' if not fname else fname
#  ext = '.jpg' if not ext else ext
#  n, bins, patches = plt.hist(x, num_bins, normed = False,
#                              facecolor = 'blue', alpha = 0.5)
#  # add a 'best fit' line
#  plt.xlabel('Bugs')
#  plt.ylabel('Frequency')
#  plt.title(r'Histogram (Median Bugs in each class)')
#
#  # Tweak spacing to prevent clipping of ylabel
#  plt.subplots_adjust(left = 0.15)
#  plt.savefig(fname + ext)
#  plt.close()

def plotCurve(x, num_bins = 10, fname = None, ext = None):

  fname = 'Untitled' if not fname else fname
  ext = '.jpg' if not ext else ext
  xlim = np.linspace(1, len(x[0]), len(x[0]))
  plt.plot(xlim, x[0], 'r', xlim, x[1], 'b');
  # add a 'best fit' line
  plt.xlabel('Rows')
  plt.ylabel('Bugs')
  # plt.title(r'Histogram (Median Bugs in each class)')

  # Tweak spacing to prevent clipping of ylabel
  plt.subplots_adjust(left = 0.15)
  plt.savefig('./_fig/' + fname + ext)
  plt.close()


