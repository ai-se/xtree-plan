from __future__ import division
from lib    import *
from demos  import *
from counts import *
from fi     import *
import re  
import sys
from settings import *
sys.dont_write_bytecode = True

def rows(file, 
          sep= The.reader.sep,
          bad= The.reader.bad):
  """Read comma-seperated rows that might be split 
  over many lines.  Finds strings that can compile 
  to nums.  Kills comments and white space."""
  n,kept = 0,""
  for line in open(file):
    now   = re.sub(bad,"",line)
    kept += now
    if kept:
      if not now[-1] == sep:
        yield n, map(atom,kept.split(sep))
        n += 1
        kept = "" 

def row(file,skip= The.reader.skip):
  "Leaps over any columns marked 'skip'."
  todo = None
  for n,line in rows(file):
    todo = todo or [col for col,name in enumerate(line) 
                    if not skip in name]
    yield n, [ line[col] for col in todo ]

## Read Headers and Rows
def table(source, rows = True, contents = row):
  t = table0(source) 
  for n,cells in contents(source):  
    if n == 0 : head(cells,t) 
    else      : body(cells,t,rows) 
  #print ">>>>", t.headers
  return t

def centroid(tbl,selections=False):
  return [h.centroid() for h in tbl.headers if (not selections or h.selected)]

## Create Table 
def table0(source):
  return Thing(
    source = source,
    depen=[], indep=[], nums =[], syms=[], 
    more =[], less =[], klass=[], headers=[], 
    _rows=[], at   ={}, patterns= The.reader.patterns)

def head(cells,t,numc=The.reader.numc):
  for col,cell in enumerate(cells):
    this   = Num if numc in cell else Sym 
    this.rank = 0
    header = this()
    header.col, header.name = col,cell
    t.at[cell] = header
    for pattern,val in t.patterns.items():
      if re.search(pattern,cell):
        where  = val(t)
        where += [header]
  return t

def body(cells,t,keep=True):
  #print "LEN?",len(t._rows)
  for n,header in enumerate(t.headers):
    cell = cells[header.col]
    #print n,"!",cell,"!"
    if not cell == The.reader.missing:
      header + cell
  if keep: 
    new = Row(cells)
    t._rows += [new] 

class Row(Thing):
  def __init__(i,cells):
    i.newId()
    i.cells = cells
    i.pos = []
    i.x0,i.y0= 0,0


def discreteTable(f,contents=lambda x: row(x)):
  rows, t = [],  table0(f)
  for n,cells in contents(f):  
    if n==0 : head(cells,t) 
    else    : rows += [cells]
  return discreteNums(t,rows)

def discreteNums(tbl,therows):
  for num in tbl.indep:
   if not num in tbl.depen:
    if isinstance(num,Num):
      for cut in  ediv(therows,
                       num=lambda x:x[num.col],
                       sym=lambda x:x[tbl.klass[0].col]):
        for row in cut._has:  
          row[num.col] = cut.range
  return clone(tbl, discrete=True, rows=therows)


def clone(tbl1,rows=[],discrete=False,keepSelections=False) :
  def ok(x):
    if x[-1]=="/":  return x
    return x.replace("$",'') if discrete else x
  tbl2= head([ok(h.name) for h in tbl1.headers],
             table0('copy of '+tbl1.source))
  if keepSelections:
    for h in tbl1.headers:
      tbl2.headers[h.col].selected = h.selected
  for cells in rows:  body(cells, tbl2, True)
  return tbl2


@demo
def tabled(f='data.dat/weather2.csv'):
  t=table(f)
  for x in  t.indep: rprintln(x)
  rprintln(t)

@demo
def tableCopied(f='data.dat/weather2.csv'):
  t0=table(f)
  t1=clone(t0)
  rprintln([t0.nums,t1.nums]); 

if __name__ == '__main__': eval(cmd())
