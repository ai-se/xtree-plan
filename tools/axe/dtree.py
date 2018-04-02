from __future__ import division
from lib import *
from demos import *
from table import *
from fi import *
from abcd import *
# from learn  import *
from settings import *
from dtree import *
import numpy as np
import sys
sys.dont_write_bytecode = True


def nl():
  print ""


def rankedFeatures(rows, t, features=None):
  features = features if features else t.indep
  klass = t.klass[0].col

  def ranked(f):
    syms, at, n = {}, {}, len(rows)
    for x in f.counts.keys():
      syms[x] = Sym()
    for row in rows:
      key = row.cells[f.col]
      val = row.cells[klass]
      syms[key] + val
      at[key] = at.get(key, []) + [row]
    e = 0
    for val in syms.values():
      if val.n:
        e += val.n / n * val.ent()
    return e, f, syms, at
  return sorted(ranked(f) for f in features)


def infogain(t, opt=The.tree):
  def norm(x):
    return (x - lo) / (hi - lo + 0.0001)
  for f in t.headers:
    f.selected = False
  lst = rankedFeatures(t._rows, t)
  tmp = [l[0] for l in lst]
  n = int(len(lst))
  n = max(n, 1)
  for _, f, _, _ in lst[:n]:
    f.selected = True
  return [f for e, f, syms, at in lst[:n]]


def tdiv1(t, rows, lvl=-1, asIs=10 ** 32, up=None, features=None, branch=[],
          f=None, val=None, opt=None):
  here = Thing(t=t, kids=[], f=f, val=val, up=up, lvl=lvl, rows=rows, modes={},
               branch=branch)
  if f and opt.debug:
    print ('|.. ' * lvl) + f.name, "=", val, len(rows)
  here.mode = classStats(here).mode()
  if lvl > 10:
    return here
  if asIs == 0:
    return here
  _, splitter, syms, splits = rankedFeatures(rows, t, features)[0]
  for key in sorted(splits.keys()):
    someRows = splits[key]
    toBe = syms[key].ent()
    if opt.variancePrune and lvl > 1 and toBe >= asIs:
      continue
    if opt.min <= len(someRows) < len(rows):
      here.kids += [tdiv1(t, someRows, lvl=lvl + 1, asIs=toBe, features=features,
                          up=here, f=splitter,
                          val=key, branch=branch + [(splitter, key)], opt=opt)]
  return here


def tdiv(tbl, rows=None, opt=The.tree):
  opt = The.tree if not opt else opt
  rows = rows or tbl._rows
  features = infogain(tbl, opt)
#  opt.min = len(rows)**0.5
  tree = tdiv1(tbl, rows, opt=opt, features=features, branch=[])
  if opt.prune:
    modes(tree)
    prune(tree)
  return tree


def modes(n):
  if not n.modes:
    n.modes = {n.mode: True}
    for kid in n.kids:
      for mode in modes(kid):
        n.modes[mode] = True
  return n.modes


def nmodes(n):
  return len(n.modes.keys())


def prune(n):
  if nmodes(n) == 1:
    n.kids = []
  for kid in n.kids:
    prune(kid)


def classStats(n):
  depen = lambda x: x.cells[n.t.klass[0].col]
  return Sym(depen(x) for x in n.rows)


def showTdiv(n, lvl=-1):
  if n.f:
    say(('|..' * lvl) + str(n.f.name) + "=" + str(n.val) +
        "\t:" + str(n.mode) + " #" + str(nmodes(n)))
  if n.kids:
    nl()
    for k in n.kids:
      showTdiv(k, lvl + 1)
  else:
    s = classStats(n)
    print ' ' + str(int(100 * s.counts[s.mode()] / len(n.rows))) + '% * ' + str(len(n.rows))


def dtnodes(tree, lvl=0):
  if tree:
    yield tree, lvl
    for kid in tree.kids:
      lvl1 = lvl
      for sub, lvl1 in dtnodes(kid, lvl1 + 1):
        yield sub, lvl1


def dtleaves(tree):
  for node, _ in dtnodes(tree):
    # print "K>", tree.kids[0].__dict__.keys()
    if not node.kids:
      yield node

# if tree:
 #   if tree.kids:
  #    for kid in tree.kids:
   #     for leaf in leaves(kid):
    #      yield leaf
    # else:
      # yield tree


def xval(tbl, m=None, n=None, opt=The.tree):
  m = m or The.tree.m
  n = n or The.tree.n
  cells = map(lambda row: opt.cells(row), tbl._rows)
  all = m * n
  for i in range(m):
    print "*" * all
    cells = shuffle(cells)
    div = len(cells) // n
    for j in range(n):
      all -= 1
      lo = j * div
      hi = lo + div
      train = clone(tbl, cells[:lo] + cells[hi:])
      test = map(Row, cells[lo:hi])
      yield test, train


def last(lst):
  return lst[-1]


def apex(test, tree, opt=The.tree):
  """apex=  leaf at end of biggest (most supported)
   branch that is selected by test in a tree"""
  def equals(val, span):
    if val == opt.missing or val == span:
      return True
    else:
      if isinstance(span, tuple):
        lo, hi = span
        return lo <= val <= hi  # <hi
      else:
        return span == val

  def apex1(cells, tree):
    found = False
    for kid in tree.kids:
      val = cells[kid.f.col]
      if equals(val, kid.val):
        for leaf in apex1(cells, kid):
          found = True
          yield leaf
    if not found:
      yield tree
  leaves = [(len(leaf.rows), leaf)
            for leaf in apex1(opt.cells(test), tree)]
  return second(last(sorted(leaves)))


def classify(test, tree, opt=The.tree):
  return apex(test, tree, opt=The.tree).mode


def improve(test, tree, opt=The.tree):
  return change(test, tree, opt.better, opt)


def degrade(test, tree, opt=The.tree):
  return change(test, tree, opt.worse, opt)


def change(test, tree, how, opt=The.tree):
  leaf1 = apex(test, tree, opt)
  new = old = leaf.mode
  if how(leaf):
    copy = opt.cells(test)[:]
    for col, val in how(leaf1).items():
      copy[col] = val
    new = classify(Row(copy), tree, opt)
  return old, new


def jumpUp(test, tree, opt=The.tree):
  return jump(test, tree, opt.better, opt)


def jumpDown(test, tree, opt=The.tree):
  return jump(test, tree, opt.worse, opt)


def jump(test, tree, how, opt=The.tree):
  toBe = asIs = apex(test, tree, opt)
  if how(asIs):
    copy = opt.cells(test)[:]
    for col, val in how(asIs).items():
      copy[col] = val
    toBe = apex(Row(copy), tree, opt)
  return asIs, toBe


def rows1(row, tbl, cells=lambda r: r.cells):
  print ""
  for h, cell in zip(tbl.headers, cells(row)):
    print h.col, ") ", h.name, cell


def snakesAndLadders(tree, train, w):
  def klass(x):
    return x.cells[train.klass[0].col]

  def l2t(l):
    return l.tbl

  def xpect(tbl):
    return tbl.klass[0].centroid()

  def score(l):
    if callable(w):
      return w(l)
    if isinstance(w, dict):
      return w[xpect(l2t(l))]
    return l
  for node in dtnodes(tree):
    node.tbl = clone(train,
                     rows=map(lambda x: x.cells, node.rows),
                     keepSelections=True)
    node.tbl.centroid = centroid(node.tbl, selections=True)
  for node1 in dtnodes(tree):
    id1 = node1._id
    node1.far = []
    node1.snake = None
    node1.worse = []
    node1.ladder = None
    node1.better = []
    for node2 in dtnodes(tree):
      # if id1 > node2._id:
      sames = overlap(node1.tbl.centroid, node2.tbl.centroid)
      node1.far += [(sames, node2)]
      # node2.far += [(sames,node1)]
  for node1 in dtnodes(tree):
    # sorted in reverse order of distance
    node1.far = sorted(node1.far,
                       key=lambda x: first(x))
    # at end of this loop, the last ladder, snakes are closest
    for _, node2 in node1.far:
      delta = prefer(node2.branch, node1.branch, key=lambda x: x.col)
      if delta:
        if score(node2) > score(node1):
          node1.ladder = node2
          node1.better = delta
        if score(node2) < score(node1):
          node1.snake = node2
          node1.worse = delta
  for node in dtnodes(tree):
    snake = node.snake._id if node.snake else None
    ladder = node.ladder._id if node.ladder else None


@demo
def tdived(file='data.dat/diabetes.csv'):
  tbl = discreteTable(file)
  # exit()
  tree = tdiv(tbl)
  showTdiv(tree)


@demo
def cross(file='data.dat/housingD.csv', rseed=1):
  def klass(test):
    return test.cells[train.klass[0].col]
  seed(rseed)
  tbl = discreteTable(file)
  n = 0
  abcd = Abcd()
  nLeaves = Num()
  nNodes = Num()
  for tests, train in xval(tbl):
    tree = tdiv(train)
    for node in dtnodes(tree):
      print node.branch
    nLeaves + len([n for n in dtleaves(tree)])
    nNodes + len([n for n in dtnodes(tree)])
    for test in tests:
      want = klass(test)
      got = classify(test, tree)
      abcd(want, got)
    exit()
  nl()
  abcd.header()
  abcd.report()
  print ":nodes", sorted(nNodes.some.all())
  print ":leaves", sorted(nLeaves.some.all())

ninf = float("-inf")


@demo
def snl(file='data.dat/poi-1.5D.csv', rseed=1, w=dict(_1=0, _0=1)):
  def klass(x):
    return x.cells[train.klass[0].col]

  def val(xxx_todo_changeme):
    (x, y) = xxx_todo_changeme
    return y if x == ninf else x
  seed(rseed)
  nl()
  print "#", file
  tbl = discreteTable(file)
  tree0 = tdiv(tbl)
  showTdiv(tree0)
  nl()
  old, better, worse = Sym(), Sym(), Sym()
  abcd1, abcd2 = Abcd(db=file, rx="where"), Abcd(db=file, rx="ranfor")
  abcd3 = Abcd(db=file, rx="logref")
  abcd4 = Abcd(db=file, rx="dt")
  abcd5 = Abcd(db=file, rx="nb")
  for tests, train in xval(tbl):
    learns(tests, train._rows,
           indep=lambda row: map(val, row.cells[:-2]),
           dep=lambda row: row.cells[-1],
           rf=abcd2,
           lg=abcd3,
           dt=abcd4,
           nb=abcd5),
    tree = tdiv(train)
    snakesAndLadders(tree, train, w)
    for test in tests:
      abcd1(actual=klass(test),
            predicted=classify(test, tree))
      a, b = improve(test, tree)
      old + a
      better + b
      _, c = degrade(test, tree)
      worse + c
  print "\n:asIs", old.counts
  print ":plan", better.counts
  print ":warn", worse.counts
  abcd1.header()
  abcd1.report()
  abcd2.report()
  abcd3.report()
  abcd4.report()
  abcd5.report()


def plot(x, y, title, xlabel, ylabel, fname):
  import matplotlib.mlab as mlab
  import matplotlib.pyplot as plt
  plt.plot(x, y)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.title(title)
  plt.subplots_adjust(left=0.15)
  plt.savefig(fname + '.jpg')
  plt.close()
# cross()
if __name__ == '__main__':
  eval(cmd())
