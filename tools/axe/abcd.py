from __future__ import division
import sys
sys.dont_write_bytecode = True


class Abcd:

  def __init__(i, db="all", rx="all", show=False):
    i.db = db
    i.rx = rx
    i.show = show
    i.yes = i.no = 0
    i.known = {}
    i.a = {}
    i.b = {}
    i.c = {}
    i.d = {}

  def __call__(i, actual=None, predicted=None):
    return i.keep(actual, predicted)

  def tell(i, actual, predict):
    i.knowns(actual)
    i.knowns(predict)
    if actual == predict:
      i.yes += 1
    else:
      i.no += 1
    for x in i.known:
      if actual == x:
        if predict == actual:
          i.d[x] += 1
        else:
          i.b[x] += 1
      else:
        if predict == x:
          i.c[x] += 1
        else:
          i.a[x] += 1

  def knowns(i, x):
    if not x in i.known:
      i.known[x] = i.a[x] = i.b[x] = i.c[x] = i.d[x] = 0.0
    i.known[x] += 1
    if (i.known[x] == 1):
      i.a[x] = i.yes + i.no

  def header(i):
    if i.show:
      print "#", ('{0:20s} {1:11s}  {2:4s}  {3:4s} {4:4s} ' +
                  '{5:4s}{6:4s} {7:3s} {8:3s} {9:3s} ' +
                  '{10:3s} {11:3s}{12:3s}            {13:30s}').format(
          "db", "rx",
          "n", "a", "b", "c", "d", "acc", "pd", "pf", "prec",
          "f", "g", "class")
      print '-' * 100

  def ask(i, raw=True):
    def p(y):
      return int(100 * y + 0.5)

    def n(y):
      return int(y)
    pd = pf = pn = prec = g = f = acc = 0
    for x in i.known:
      a = i.a[x]
      b = i.b[x]
      c = i.c[x]
      d = i.d[x]
      if (b + d):
        pd = d / (b + d)
      if (a + c):
        pf = c / (a + c)
      if (a + c):
        pn = (b + d) / (a + c)
      if (c + d):
        prec = d / (c + d)
      if (1 - pf + pd):
        g = 2 * (1 - pf) * pd / (1 - pf + pd)
      if (prec + pd):
        f = 2 * prec * pd / (prec + pd)
      if (i.yes + i.no):
        acc = i.yes / (i.yes + i.no)
      if i.show:
        print "#", ('{0:20s} {1:10s} {2:4d} {3:4d} {4:4d} ' +
                    '{5:4d} {6:4d} {7:4d} {8:3d} {9:3d} ' +
                    '{10:3d} {11:3d} {12:3d}           {13:30s}').format(i.db,
                                                                         i.rx, n(b + d), n(a), n(b), n(c), n(d),
                                                                         p(acc), p(pd), p(pf), p(prec), p(f), p(g), str(x) + ' Bugs')
      return [p(acc), p(pd), p(1 - pf), p(prec), p(f), p(g)]


def _Abcd(before=None, after=None, show=False):
  abcd = Abcd(db='Actual', rx='Predicted', show=show)
  train = before
  test = after
  # random.shuffle(test)
  for actual, predicted in zip(train, test):
    abcd.tell(actual, predicted)
  if show:
    abcd.header()
  return abcd.ask()

"""
output:
 $ python ABCD.py
# db                   rx           n     a    b    c   d    acc pd  pf  prec f  g  class
----------------------------------------------------------------------------------------------------
# randomIn             jiggle       22    0    5    5   17   63  77 100  77  77   0 a
# randomIn             jiggle        5   17    5    5    0   63   0  23   0  77   0 b
"""

if __name__ == "__main__":
  _Abcd()
