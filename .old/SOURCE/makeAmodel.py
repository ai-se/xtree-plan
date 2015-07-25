import sys
from table import *
import smote


def say(text):
  sys.stdout.write(str(text)), sys.stdout.write(' ')


class o:

  def __init__(self, **d):
    self.update(**d)

  def update(self, **d):
    self.__dict__.update(**d)
    return self


class makeAModel(object):

  def __init__(self):
    self.seen = []
    self.translate = {}
    pass

  def data(self, indep=[], less=[], more=[], _rows=[]):
    nindep = len(indep)
    ndep = len(less) + len(more)
    m = o(lo={}, hi={}, w={},
          eval=lambda m, it: True,
          _rows=[o(cells=r, score=0, scored=False,
                   x0=None, y0=None)
                 for r in _rows],
          names=indep + less + more)
    m.decisions = [x for x in range(nindep)]
    m.objectives = [nindep + x - 1 for x in range(ndep)]
    for k, indx in enumerate(m.decisions):
      for l in m.objectives:
        if k == l:
          m.decisions.pop(indx)
    m.cols = m.decisions + m.objectives
    for x in m.decisions:
      m.w[x] = 1
    for y, _ in enumerate(less):
      m.w[x + y] = -1
    for z, _ in enumerate(more):
      m.w[x + y + z] = 1
    for x in m.cols:
      all = sorted(row.cells[x] for row in m._rows)
      m.lo[x] = all[0]
      m.hi[x] = all[-1]
    return m

  def str2num(self, tbl):
    P = 1
    for row in tbl._rows:
      for k in row.cells:
        if not k in self.seen and isinstance(k, str):
          self.seen.append(k)
          self.translate.update({k: P}) if isinstance(k, str) \
              else self.translate.update({k: k})
          P += 1

  def csv2py(self, filename, _smote=False, duplicate=False):
    "Convert a csv file to a model file"
    tbl = table(filename)
    if _smote:
      tbl = smote.SMOTE(
          tbl,
          atleast=50,
          atmost=101,
          bugIndx=1,
          resample=duplicate)
    self.str2num(tbl)
    tonum = lambda x: self.translate[x] if isinstance(x, str) else x

    """ There's a bug in table.py that doesn't separate dependent and independent
      Variable. The following, badly written, piece of code corrects for it...
  """
    for indx, k in enumerate(tbl.indep):
      for l in tbl.depen:
        if k.name == l.name:
          tbl.indep.pop(indx)

    return self.data(indep=[i.name for i in tbl.indep],
                     less=[i.name for i in tbl.depen],
                     _rows=map(lambda x: [tonum(xx) for xx in x.cells],
                               tbl._rows))


def _makeamodel(file=None):
  makeaModel = makeAModel()
  m = makeaModel.csv2py(file)
  return m

if __name__ == '__main__':
  pass
