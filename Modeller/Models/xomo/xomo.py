import sys, pdb
sys.dont_write_bytecode=True

def demo(f=None,demos=[]): 
  if f: demos.append(f); return f
  for d in demos: 
    print '\n--|',d.func_name,'|','-'*40,'\n',d.__doc__,'\n'
    d()

def test(f=None,tests=[]): 
  if f: tests.append(f); return f
  ok=no=0
  for t in tests: 
    print "# ",t.func_name + ': ',t.__doc__
    for n,(want,got) in  enumerate(t()):
      if want == got:
        ok += 1; print "PASSED:",t.func_name,n+1
      else:
        no += 1; print "FAILED:",t.func_name,n+1
  if tests:
    print '\n# totals: %s%% PASSED' % round(100*ok/(ok+no))

@test
def tested1():
  "Test functions return lists of (want,got) pairs"
  return [(1,1),(0,1),(2,2),(1,0)]

@test
def tested2():
  "Test function can return one pair"
  return [(1,1)]
import random


class Deep(dict) :
  def __getitem__(self,x) :
    if x in self: return self.get(x)
    new = self[x] = Deep()
    return new
  def push(self,k,v) :
    all = self[k] if k in self else []
    all.append(v)
    self[k]=all
    return all
  def at(self,lst,default=None) :
    here=self
    for key in lst:
      if not key in here:
        return default
      here = here[key]
    return here
  def inc(self,k,n=1):
    new = (self[k] if k in self else 0) + n
    self[k] = new
    return new 

@demo
def _deeped() :
  "Auto-generation of  nested dictionaries"
  d=Deep()
  d[1][2][3] = 22
  d[1][2][4] = 44
  print d


class Sample(object): 
  def one(o)   : pass
  def ready(o) : pass
  def some(o,max=100):
    o.ready()
    while True:
      max -= 1
      if max < 0 : break
      yield o.one()

class Some(Sample):
  def __init__(o,txt):
    o.txt=txt
  def bias(o,lst,scale=2):
    o.all = [(n**scale,x) for n,x in enumerate(lst)]
    return o
  def ready(o):
    o.all = sorted(o.all,key=lambda x: x[0])
    o.total = 0
    for n,x in o.all: o.total += n
  def one(o):
    chosen = random.uniform(0,o.total)
    count  = 0
    for n,x in o.all:
      count += n
      if count > chosen: return x   
  
@demo 
def somed1(): 
  "Biased list"
  somed0(Some("xx").bias([x for x in xrange(0,25)]),0.2)

def somed0(r,shrink=2,n=1000):
   def show(k,v) :
     return str(k).rjust(10)+ ' : '+ \
            '*'*int(v)+ ' [%3.2f]%%'%int(shrink*v)
   all = Deep()
   random.seed(1)
   for x in r.some(max=n): all.inc(round(x,1)) 
   print ""
   #order = sorted([int(x) for x in all.keys()])
   order = sorted(all.keys())
   for k in order:
     v = all[k]
     print "DEBUG: " + str(v) + " " + str(k)
     # for k,v in  sorted(all.items(),key=all.get):
     print show(k,100.00/shrink*v/n) 

class Runiform(Sample):
   def one(o): 
     return o.final(random.uniform(o.lo,o.hi))
   def __init__(o,txt,lo,hi,final=float):
     o.txt= txt; o.lo= lo; o.hi= hi; o.final= final

@demo
def somed1():
  "Uniform 1 to 5" 
  somed0(Runiform("xx",1,5,int),0.5)

class Rtriangle(Sample):
   def one(o): 
     return o.final(random.triangular(o.lo,o.hi,o.mode))
   def __init__(o,txt,lo,hi,mode,final=float):
     o.txt=txt; o.lo=lo; o.hi=hi; o.mode=mode; o.final=final

@demo
def somed2(): 
  "Triangle min,max,mode = 0,20,4"
  somed0(Rtriangle("xx",0,20,4,int),0.25)

class Rgamma(Sample):
   def one(o): 
     return o.final(random.gammavariate(o.a,o.b))
   def __init__(o,txt,a,b,final=float):
     o.txt= txt; o.a= a; o.b= b; o.final=final

@demo
def somed3(): 
  "Gamma a,b = 5,1"
  somed0(Rgamma("xx",6,1,int),0.33)

class Rexpo(Sample):
   def one(o): 
     return o.final(random.expovariate(o.lambd))
   def __init__(o,txt,lambd,final=float):
     o.txt= txt; o.lambd= lambd; o.final= final

@demo
def somed4(): 
  "Lambda, decay constant=0.7" 
  somed0(Rexpo("xx",0.7,int),1)

class Rgauss(Sample):
   def one(o): 
     return o.final(random.gauss(o.mu,o.sigma))
   def __init__(o,txt,mu,sigma,final=float):
     o.txt= txt; o.mu= mu; o.sigma= sigma; o.final= final

@demo
def somed5(): 
  "Guassian, mean=20, sigma=2"
  somed0(Rgauss("xx",20,2,int),0.5)

class Rsteps(Sample):
  def __init__(o,txt,bins=7,final=float):
    o.txt= txt; o.bins= bins; o.final= final
    o.all=[]; o.stale=False
  def bias(o,lst): 
    o.stale=True
    o.all=lst
  def put(o,x):
    o.stale=True
    o.all.append(x)
  def ready(o):
    if o.stale:
      o.all = sorted(o.all)
      split = max(1, int(len(o.all)/o.bins))
      o.all = [o.all[int(bin*split)] for bin in range(o.bins)]
    o.stale=False
  def __sub__(o1,o2):
    o1.ready(); o2.ready()
    diff = sum1 = sum2 = 0.0
    for n1 in o1.all: sum1 += n1
    for n2 in o2.all: sum2 += n2
    for n1,n2 in zip(o1.all,o2.all) : 
      print n1,n2,sum1,sum2
      diff += (n1/sum1 - n2/sum2)
    return 100*diff
  def one(o):
    o.ready()
    n = random.randint(1,o.bins-1) 
    return o.final(random.uniform( o.all[n-1], o.all[n]))

@demo 
def somed6(): 
  "Divide data into 5 steps"
  lst = [x for x in xrange(0,33)]
  somed0(Rsteps("xx",7,int).bias(lst))
import random

def any(l):
   return l[random.randint(0,len(l)-1)]

def chunks(l, n):
  "Divide 'l' into sub-lists of length 'n'."
  return [l[i:i+n] for i in range(0, len(l), n)]

def often(seq,max=100,
          item   = lambda x: x,
          weight = lambda x: x.priority) :
    total = 0
    for x in seq: total += weight(x)
    while True:
        max -= 1
        if max < 0 : break
        chosen = random.uniform(0, total)
        count  = 0
        for x in seq:
            count += weight(x)
            if count > chosen:
                yield item(x)
                break

@test
def oftened():
  "select, with bias, from a space"
  def left(x) : return x[0]
  def right(x): return x[1]
  counts = Deep()
  random.seed(1)
  for x in often([("a",10),("b",20),("c",40),("d",80)],
                 max=1000,item=left,weight=right):
    counts.inc(x)
  return [(counts,{'a': 67, 'c': 265, 'b': 113, 'd': 555})]
import random

class X2Y(object):
  def x(o): pass
  def y(o,x) : pass
  def xy(o):
    x = o.x()
    y = o.y(x)
    return x,y

class Range(X2Y):
  def __init__(o,name,min,max,final=float,wild=False):
    o.txt    = name
    o.wild   = wild
    o.update(min,max,final)
  def update(o,min,max,final=float,m=None):
    o.min = min
    o.max = max
    o.sample = Runiform(o.txt,min,max,final)
    if m:
      m.all[o.txt] = o
  def x(o):
    return o.sample.one()

class Model(X2Y):
  def __init__(o):
    o.all = {}
    for i in o.about(): 
      o.all[i.txt] = i
  def x(o):
    out = {}
    for what,thing in o.all.items(): out[what] = thing.x()
    return out
  def about(o): pass


#################
##--SCED-RISK--##
#################

def totalRisk(project, risktable):
  _d = 3.73
  return (sced_risk(project, risktable) +
          prod_risk(project, risktable) +
          pers_risk(project, risktable) +
          proc_risk(project, risktable) +
          plat_risk(project, risktable) +
          reus_risk(project, risktable)) / _d
          

def getRisk(a, b, project, risks):
  if (a in project) and (b in project):
    _aval = int(project[a]); #print(_aval)
    _bval = int(project[b]); #print(_bval)
    try:
      _rt = risks[a, b]
      _rt = _rt.split(",")          # split table
      _rt = _rt[_bval-1]            # grab line
      _rt = _rt.split()             # split line
      return float(_rt[_aval-1])    # find index
    except (KeyError, IndexError):
      return 0
  else: return 0


def sced_risk(proj, risks):
  _a = 'sced'
  return(sum(map(lambda x : getRisk(_a,x,proj,risks),
             str.split("rely time pvol tool acap " 
                       "aexp pcap plex ltex pmat"))))


def prod_risk(proj, risks):
  return(sum(map(lambda (x,y) : getRisk(x,y,
                                        proj,risks),
                 [("rely","acap"), ("rely","pcap"),
                  ("cplx","acap"), ("cplx","pcap"),
                  ("cplx","tool"), ("rely","pmat"),
                  ("sced","cplx"), ("sced","rely"),
                  ("sced","time"), ("ruse","aexp"),
                  ("ruse","ltex")])))


def pers_risk(proj, risks):
  return(sum(map(lambda (x,y) : getRisk(x,y,
                                        proj,risks),
                 [("pmat","acap"), ("stor","acap"),
                  ("time","acap"), ("tool","acap"),
                  ("tool","pcap"), ("ruse","aexp"),
                  ("ruse","ltex"), ("pmat","pcap"),
                  ("stor","pcap"), ("time","pcap"),
                  ("ltex","pcap"), ("pvol","plex"),
                  ("sced","acap"), ("sced","aexp"),
                  ("sced","pcap"), ("sced","plex"),
                  ("sced","ltex"), ("rely","acap"),
                  ("rely","pcap"), ("cplx","acap"),
                  ("cplx","pcap"), ("team","aexp")
                  ])))


def proc_risk(proj, risks):
  return(sum(map(lambda (x,y) : getRisk(x,y,
                                        proj,risks),
                 [("tool","pmat"), ("time","tool"),
                  ("team","aexp"), ("team","sced"), 
                  ("team","site"), ("sced","tool"), 
                  ("sced","pmat"), ("cplx","tool"), 
                  ("pmat","acap"), ("tool","acap"), 
                  ("tool","pcap"), ("pmat","pcap")
                  ])))


def plat_risk(proj, risks):
  return(sum(map(lambda (x,y) : getRisk(x,y,
                                        proj,risks),
                 [("sced","time"), ("sced","pvol"),
                  ("stor","acap"), ("time","acap"),
                  ("stor","pcap"), ("pvol","plex"),
                  ("time","tool")])))


def reus_risk(project, risktable):
  return(getRisk('ruse','aexp',project,risktable) +
         getRisk('ruse','ltex',project,risktable))


#############

def readRisks(risktable):
  risktable['sced','rely'] = ("0 0 0 1 2 0,"
                              "0 0 0 0 1 0,"
                              "0 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "0 0 0 0 0 0")
  risktable['sced','cplx'] = ("0 0 0 1 2 4,"
                              "0 0 0 0 1 2,"
                              "0 0 0 0 0 1,"
                              "0 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "0 0 0 0 0 0")
  risktable['sced','time'] = ("0 0 0 1 2 4,"
                              "0 0 0 0 1 2,"
                              "0 0 0 0 0 1,"
                              "0 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "0 0 0 0 0 0")
  risktable['sced','pvol'] = ("0 0 0 1 2 0,"
                              "0 0 0 0 1 0,"
                              "0 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "0 0 0 0 0 0")
  risktable['sced','tool'] = ("2 1 0 0 0 0,"
                              "1 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "0 0 0 0 0 0")
  risktable['sced','pexp'] = ("4 2 1 0 0 0,"
                              "2 1 0 0 0 0,"
                              "1 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "0 0 0 0 0 0")
  risktable['sced','pcap'] = ("4 2 1 0 0 0,"
                              "2 1 0 0 0 0,"
                              "1 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "0 0 0 0 0 0")
  risktable['sced','aexp'] = ("4 2 1 0 0 0,"
                              "2 1 0 0 0 0,"
                              "1 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "0 0 0 0 0 0")
  risktable['sced','acap'] = ("4 2 1 0 0 0,"
                              "2 1 0 0 0 0,"
                              "1 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "0 0 0 0 0 0")
  risktable['sced','ltex'] = ("2 1 0 0 0 0,"
                              "1 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "0 0 0 0 0 0")
  risktable['sced','pmat'] = ("2 1 0 0 0 0,"
                              "1 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "0 0 0 0 0 0")
  risktable['rely','acap'] = ("0 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "1 0 0 0 0 0,"
                              "2 1 0 0 0 0,"
                              "4 2 1 0 0 0,"
                              "0 0 0 0 0 0")
  risktable['rely','pcap'] = ("0 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "1 0 0 0 0 0,"
                              "2 1 0 0 0 0,"
                              "4 2 1 0 0 0,"
                              "0 0 0 0 0 0")
  risktable['cplx','acap'] = ("0 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "1 0 0 0 0 0,"
                              "2 1 0 0 0 0,"
                              "4 2 1 0 0 0")
  risktable['cplx','pcap'] = ("0 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "1 0 0 0 0 0,"
                              "2 1 0 0 0 0,"
                              "4 2 1 0 0 0")
  risktable['cplx','tool'] = ("0 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "1 0 0 0 0 0,"
                              "2 1 0 0 0 0,"
                              "4 2 1 0 0 0")
  risktable['rely','pmat'] = ("0 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "1 0 0 0 0 0,"
                              "2 1 0 0 0 0,"
                              "4 2 1 0 0 0,"
                              "0 0 0 0 0 0")
  risktable['pmat','acap'] = ("2 1 0 0 0 0,"
                              "1 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "0 0 0 0 0 0")
  risktable['stor','acap'] = ("0 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "1 0 0 0 0 0,"
                              "2 1 0 0 0 0,"
                              "4 2 1 0 0 0")
  risktable['time','acap'] = ("0 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "1 0 0 0 0 0,"
                              "2 1 0 0 0 0,"
                              "4 2 1 0 0 0")
  risktable['tool','acap'] = ("2 1 0 0 0 0,"
                              "1 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "0 0 0 0 0 0")
  risktable['tool','pcap'] = ("2 1 0 0 0 0,"
                              "1 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "0 0 0 0 0 0")
  risktable['ruse','aexp'] = ("0 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "1 0 0 0 0 0,"
                              "2 1 0 0 0 0,"
                              "4 2 1 0 0 0")
  risktable['ruse','ltex'] = ("0 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "1 0 0 0 0 0,"
                              "2 1 0 0 0 0,"
                              "4 2 1 0 0 0")
  risktable['pmat','pcap'] = ("0 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "1 0 0 0 0 0,"
                              "2 1 0 0 0 0,"
                              "4 2 1 0 0 0")
  risktable['stor','pcap'] = ("0 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "1 0 0 0 0 0,"
                              "2 1 0 0 0 0,"
                              "4 2 1 0 0 0")
  risktable['time','pcap'] = ("0 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "1 0 0 0 0 0,"
                              "2 1 0 0 0 0,"
                              "4 2 1 0 0 0")
  risktable['ltex','pcap'] = ("4 2 1 0 0 0,"
                              "2 1 0 0 0 0,"
                              "1 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "0 0 0 0 0 0")
  risktable['pvol','pexp'] = ("0 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "1 0 0 0 0 0,"
                              "2 1 0 0 0 0,"
                              "0 0 0 0 0 0")
  risktable['tool','pmat'] = ("2 1 0 0 0 0,"
                              "1 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "0 0 0 0 0 0")
  risktable['time','tool'] = ("0 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "1 0 0 0 0 0,"
                              "2 1 0 0 0 0")
  risktable['team','aexp'] = ("2 1 0 0 0 0,"
                              "1 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "0 0 0 0 0 0")
  risktable['team','sced'] = ("2 1 0 0 0 0,"
                              "1 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "0 0 0 0 0 0")
  risktable['team','site'] = ("2 1 0 0 0 0,"
                              "1 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "0 0 0 0 0 0,"
                              "0 0 0 0 0 0")


#############
import math



# CoQualMo Calibration Mods:
DefectIntroReqsNeg = ['pmat','prec','resl','team',
                      'acap','aexp','docu','ltex', 
                      'pcon','plex','rely','sced',
                      'site','tool']
DefectIntroDesignNeg = ['pmat','prec','resl','team',
                        'acap','aexp','docu','ltex', 
                        'pcon','plex','rely','sced',
                        'site','tool','pcap']
DefectIntroCodingNeg = ['pmat','prec','resl','team',
                        'acap','aexp','docu','ltex', 
                        'pcon','plex','rely','sced',
                        'site','tool','pcap']
DefectIntroReqsPos = ['cplx','data','pvol',
                      'ruse','stor','time']
DefectIntroDesignPos = ['cplx','data','pvol',
                        'ruse','stor','time']
DefectIntroCodingPos = ['cplx','data','pvol',
                        'ruse','stor','time']
DefectIntroReqsIgnore = ["flex","pcap"]
DefectIntroDesignIgnore = ["flex"]
DefectIntroCodingIgnore = ["flex"]
DefectRemovers = ["aa","etat","pr"]


class Cocomo(Model):
  def __init__(o, *args, **kwargs):
    
    o.bounds = {}
    _s = "flex pmat prec resl team"
    o.scaleFactors = _s.split()
    _s = ("acap aexp cplx data docu ltex "
          "pcap pcon plex pvol rely ruse "
          "sced site stor time tool")
    o.effortMultipliers = _s.split()
    _s = "aa etat pr"
    o.defectRemovers = _s.split()
    # Open file of constraints (?):
    for _a in args:
      if isinstance(_a, basestring):
        try:
          _acc = {}
          with open(_a,'r') as infile:
            for line in infile:
              line = line.split()
              if line[0] == '@project':
                o.proj = line[1]
              elif line[0] == '@discrete':
                if line[1][0] == '?':
                  _attr = line[1][1:]
                  _lo   = float(line[2])
                  _hi   = float(line[len(line)-1])
                else:
                  _attr = line[1]
                  _lo = _hi = int(line[2])
                _acc[_attr] = _lo,_hi
          # Overwrite file constraints w/kwargs:
          kwargs = dict(_acc.items() + kwargs.items())
          break
        except:
          pass
          #print "Input file [", _a, "] not readable"
    # Read constraints from kwargs:
    for _key,_val in kwargs.items():
#      print _key, _val
      if ((_key in o.scaleFactors) or
          (_key in o.effortMultipliers) or
          (_key in o.defectRemovers) or
          (_key in ["kloc","b"])):
        if isinstance(_val, tuple):
          _lo,_hi = _val
        else:
          _lo = _hi = _val
        o.bounds[str(_key)] = _lo,_hi
    # Parent init:
    super(o.__class__,o).__init__()
    

  def say(o,x,a,b,kloc,sum,prod,exp,
          effort,  # o1\o2,o3,o4
          months="nc",defects="nc",risks="nc"):
    for i,j in x.items():  
      if i=="kloc": print i,j
      else: print i,j,str(o.all[i].y(j))
    print (":a",a,":b",b,":kloc",kloc,":exp",exp,
           ":sum",sum,":prod",prod, "effort",effort,
           "months",months,"defects",defects,
           "risks",risks)
  def sumSfs(o,x,out=0,reset=False): 
    for i in o.scaleFactors:
      out += o.all[i].y(x[i],reset)
    return out

  def prodEms(o,x,out=1,reset=False):
    for i in o.effortMultipliers:
       out *= o.all[i].y(x[i],reset) #changed_nave
    return out

  def xy(o,verbose=False):
    x     = o.x()
    a     = x["b"]  # a little tricky... "a" is the x of "b"
    b     = o.all["b"].y(a,reset=True)
    kloc  = o.all["kloc"].x()
    sum   = o.sumSfs(x,reset=True)
    prod  = o.prodEms(x,reset=True)
    exp   = b + 0.01 * sum
    effort = a*(kloc**exp)*prod
    if verbose: o.say(x,a,b,kloc,sum,prod,exp,effort)
    return x,effort 

  def xys(o,verbose=False,olist=False):
    x     = o.x()
    a     = x["b"]
    b     = o.all["b"].y(a,reset=True)
    kloc = x["kloc"]
    sum   = o.sumSfs(x,reset=True)
    prod  = o.prodEms(x,reset=True)
    exp   = b + 0.01 * sum
    effort  = o.effort_calc(x, a, b, exp, sum, prod)
    months  = o.month_calc(x, effort, sum, prod)
    defects = o.defect_calc(x)
    risks   = o.risk_calc(x)
    if verbose: o.say(x,a,b,kloc,sum,prod,exp,
                      effort,months,defects,risks)
    if olist:
      return [effort,months,defects,risks]
    else:
      return x,effort,months,defects,risks

  def trials(o,n=500,out="out.csv",verbose=True,write=False):
    import csv
    keys = []
    _efforts = []
    _months = []
    _defects = []
    _risks = []
    _first = 0
    rows = []
    with open(out,'w') as csv_file:
      if write: csv_wri = csv.writer(csv_file)
      for _i in range(0,n):
        x    = o.x()
        if _i == 0:
          for _k,_ in x.iteritems():
            if _first == 0:
              keys.append('$'+str(_k))
              _first = 1
            else:
              keys.append('$'+str(_k)) #changed_nave
          keys.extend(["-effort","-months",
                       "-defects","-risks"])
          if write: csv_wri.writerows([keys])
        a    = x["b"]
        b    = o.all["b"].y(a,reset=True)
        kloc = x["kloc"]
        sum  = o.sumSfs(x,reset=True)
        prod = o.prodEms(x,reset=True)
        exp  = b + 0.01 * sum
        effort  = o.effort_calc(x,a,b,exp,sum,prod)
        months  = o.month_calc(x,effort,sum,prod)
        defects = o.defect_calc(x)
        risks   = o.risk_calc(x)
        _efforts.append(effort)
        _months.append(months)
        _defects.append(defects)
        _risks.append(risks)
        vals = []
        for _,_v in x.iteritems():
          vals.append(_v)
        vals.extend([effort,months,defects,risks])
        if write: csv_wri.writerows([vals])
        rows.append(vals)

    if verbose:
      _effSum = math.fsum(_efforts)
      _mosSum = math.fsum(_months)
      _defSum = math.fsum(_defects)
      _rskSum = math.fsum(_risks)
      _effMean = _effSum/n
      _mosMean = _mosSum/n
      _defMean = _defSum/n
      _rskMean = _rskSum/n
      _effSD = pow( math.fsum(map(lambda x: pow(x-_effMean,2),_efforts))/n, 0.5)
      _mosSD = pow( math.fsum(map(lambda x: pow(x-_mosMean,2),_months))/n, 0.5)
      _defSD = pow( math.fsum(map(lambda x: pow(x-_defMean,2),_defects))/n, 0.5)
      _rskSD = pow( math.fsum(map(lambda x: pow(x-_rskMean,2),_risks))/n, 0.5)
      _efforts.sort()
      _months.sort()
      _defects.sort()
      _risks.sort()
      print "Means:"
      print "\tEff:",_effMean,"\n\tMos:",_mosMean,"\n\tDef:",_defMean,"\n\tRsk:",_rskMean
      print ""
      print "Standard Deviations:"
      print "\tEff:",_effSD,"\n\tMos:",_mosSD,"\n\tDef:",_defSD,"\n\tRsk:",_rskSD
      print ""
      print "Quartile Bounds (25/50/75):"
      print "\tEff:", _efforts[int(.25*n)],"\t",\
        _efforts[int(.5*n)],"\t",\
        _efforts[int(.75*n)],  \
        "\n\tMos:", _months[int(.25*n)],"\t",\
        _months[int(.5*n)],"\t",\
        _months[int(.75*n)],  \
        "\n\tDef:", _defects[int(.25*n)],"\t",\
        _defects[int(.5*n)] ,"\t",\
        _defects[int(.75*n)],  \
        "\n\tRsk:", _risks[int(.25*n)],"\t",\
        _risks[int(.5*n)],"\t",\
        _risks[int(.75*n)]
    return keys,rows


  def about(o):
    def dr(what, lo=1,hi=6) : return Dr(what,lo,hi)
    def sf(what, lo=1,hi=5) : return Sf(what,lo,hi)
    def emn(what,lo=1,hi=5) : return Emn(what,lo,hi)
    def emp(what,lo=1,hi=5) : return Emp(what,lo,hi) 
    _rtn = []
    # kloc:
    if "kloc" in o.bounds:
      _lo,_hi = o.bounds["kloc"]
    else: _lo,_hi = 2,1000
    _rtn.append( Range("kloc",_lo,_hi) )
    # b (becomes 'a')
    if "b" in o.bounds:
      _lo,_hi = o.bounds["b"]
      _rtn.append( B("b",_lo,_hi) )
    else: 
      _lo,_hi = 3,10
      _rtn.append( B("b",3,10,wild=True) )
    # Defect Removers:
    for _dr in ["aa",
                "etat",
                "pr"  ]:
      if _dr in o.bounds:
        _lo,_hi = o.bounds[_dr]
        _rtn.append( dr(_dr,_lo,_hi) )
      else:
        _rtn.append( dr(_dr) )
    # Scale Factors:
    for _sf in ["prec", "flex",
                "resl", "team",
                "pmat"        ]:
      if _sf in o.bounds:
        _lo,_hi = o.bounds[_sf]
        _rtn.append( sf(_sf,_lo,_hi) )
      else:
        _rtn.append( sf(_sf) )
    # Effort Multipliers, Positive Slope
    for _emp, _rng in [ ( "rely", (1,5) ),
                        ( "data", (2,5) ),
                        ( "cplx", (1,6) ),
                        ( "ruse", (2,6) ),
                        ( "docu", (1,5) ),
                        ( "time", (3,6) ),
                        ( "stor", (3,6) ),
                        ( "pvol", (2,5) )]:
      if _emp in o.bounds:
        _lo,_hi = o.bounds[_emp]
      else:
        _lo,_hi = _rng
      _rtn.append( emp(_emp,_lo,_hi) )
    # Effort Multipliers, Negative Slope
    for _emn in ["acap", "pcap",
                 "pcon", "aexp",
                 "plex", "ltex",
                 "tool", "site",
                 "sced"        ]:
      if _emn in o.bounds:
        _lo,_hi = o.bounds[_emn]
      else:
        if _emn == "site":
          _hi = 6  # Special case
        else:
          _hi = 5  # (Default)
        _lo = 1
      _rtn.append( emn(_emn,_lo,_hi) )
    return _rtn


  def effort_calc(o, x, 
                  a=-1, b=-1, exp=-1,
                  sum=-1, prod=-1):
    if    a == -1: a    = x["b"]
    if    b == -1: b    = o.all["b"].y(a)
    if  sum == -1: sum  = o.sumSfs(x)
    if  exp == -1: exp  = b + 0.01 * sum
    if prod == -1: prod = o.prodEms(x)
    return a*abs(x["kloc"])**exp*prod

  def month_calc(o, x, effort, 
                 sum=-1, prod=-1):
    if  sum == -1: sum  = o.sumSfs(x)
    if prod == -1: prod = o.prodEms(x)
    _c    = 3.67
    _d    = 0.28
    _sced = int(x["sced"])
    _scedPercent = 0
    if (_sced == 1):
      _scedPercent = 75
    elif (_sced == 2):
      _scedPercent = 85
    elif (_sced == 3):
      _scedPercent = 100
    elif (_sced == 4):
      _scedPercent = 130
    elif (_sced == 5):
      _scedPercent = 160
    _pmNs = (effort / 
             float(o.all["sced"].y(x["sced"])))
    _elessb = 0.01 * sum
    _f = _d + (0.2 * _elessb)
    return _c * pow(abs(_pmNs),_f) * (_scedPercent/100.0)

  def defect_calc(o, x):
    return (o.defects("requirements", x) +
            o.defects("design", x) +
            o.defects("code", x))

  def defects(o, dtype, x):  
    _ksloc = float(x["kloc"])
    _introduced = 0
    if (dtype == "requirements"): 
      _introduced = (10 * _ksloc * 
                     o.defectsIntroduced(dtype,x))
    elif (dtype == "design"):
      _introduced = (20 * _ksloc * 
                     o.defectsIntroduced(dtype,x))
    elif (dtype == "code"):
      _introduced = (30 * _ksloc * 
                     o.defectsIntroduced(dtype,x))
    _percentRemoved = o.defectsRemovedRatio(dtype,x)
    return _introduced * _percentRemoved

  def defectsRemovedRatio(o, dtype,x):
    _product = 1
    for _key in o.defectRemovers:
      if _key in x:
        if (dtype == "requirements"):
          _product *= (1 - float(
            o.all[
                _key
            ].calibs.defectRemovalReqs.y(x[_key])
          ))
        elif (dtype == "design"):
          _product *= (1 - float(
              o.all[
                _key
              ].calibs.defectRemovalDesign.y(x[_key])
          ))
        elif (dtype == "code"):
          _product *= (1 - float(
             o.all[
                _key
             ].calibs.defectRemovalCoding.y(x[_key])
          ))
    return _product

  def totalDefectsIntroduced(o,x):
    _ksloc = x["kloc"]
    return (10 * _ksloc * 
            o.defectsIntroduced(
              "requirements", x) +
            20 * _ksloc *
            o.defectsIntroduced(
              "design", x) +  
            30 * _ksloc *
            o.defectsIntroduced(
              "code", x))  

  def defectsIntroduced(o, dtype, x):
    _product = 1
    for _key in o.scaleFactors:
      if _key in x:
        if (dtype == "requirements"):
          _product *= float(
            o.all[
              _key
            ].calibs.defectIntroReqs.y(x[_key])
        )
        elif (dtype == "design"):
          _product *= float(
            o.all[
              _key
            ].calibs.defectIntroDesign.y(x[_key])
        )
        elif (dtype == "code"):
          _product *= float(
            o.all[
              _key
            ].calibs.defectIntroCoding.y(x[_key])
        )
      else:
        print ("Err: " + _key +
               " not defined in source input")
    for _key in o.effortMultipliers:
      if _key in x:
        if (dtype == "requirements"):
          _product *= float(
            o.all[
              _key
            ].calibs.defectIntroReqs.y(x[_key])
        )
        elif (dtype == "design"):
          _product *= float(
            o.all[
              _key
            ].calibs.defectIntroDesign.y(x[_key])
        )
        elif (dtype == "code"):
          _product *= float(
            o.all[
              _key
            ].calibs.defectIntroCoding.y(x[_key])
        )
      else:
        print ("Err: " + _key +
               " not defined in source input")
    return _product

  def risk_calc(o, x):
    rt = {}
    readRisks(rt)
    return totalRisk(x, rt) 



class Calibrations():
  """CoQualMo calibration settings for
   a given CoCoMo attribute"""
  def __init__(o,txt):
    # Requirements:
    if   txt in DefectIntroReqsPos:
      o.defectIntroReqs = Calib('Intro',
                                'Reqs',
                                 1 )
    elif txt in DefectIntroReqsNeg:
      o.defectIntroReqs = Calib('Intro',
                                'Reqs',
                                -1 )
    elif txt in DefectIntroReqsIgnore:
      o.defectIntroReqs = Calib('Intro',
                                'Reqs',
                                 0 )
    else: o.defectIntroReqs = None
    # Design:
    if   txt in DefectIntroDesignPos:
      o.defectIntroDesign = Calib('Intro',
                                  'Design',
                                   1 )
    elif txt in DefectIntroDesignNeg:
      o.defectIntroDesign = Calib('Intro',
                                  'Design',
                                  -1 )
    elif txt in DefectIntroDesignIgnore:
      o.defectIntroDesign = Calib('Intro',
                                  'Design',
                                   0 )
    else: o.defectIntroDesign = None
    # Coding:
    if   txt in DefectIntroCodingPos:
      o.defectIntroCoding = Calib('Intro',
                                  'Coding',
                                   1 )
    elif txt in DefectIntroCodingNeg:
      o.defectIntroCoding = Calib('Intro',
                                  'Coding',
                                  -1 )
    elif txt in DefectIntroCodingIgnore:
      o.defectIntroCoding = Calib('Intro',
                                  'Coding',
                                   0 )
    else: o.defectIntroCoding = None
    # Removal:
    if   txt in DefectRemovers:
      o.defectRemovalReqs   = Calib('Removal',
                                    'Reqs',
                                    0)
      o.defectRemovalDesign = Calib('Removal',
                                    'Design',
                                    0)
      o.defectRemovalCoding = Calib('Removal',
                                    'Coding',
                                    0)


class Calib():
  """CoQualMo calibration data generator"""
  def __init__(o, phase, category, sign):
    o.phase = phase        # Intro/Removal
    o.category = category  # Reqs/Dsgn/Code
    o.sign = sign          # Slope Pos/Neg
    o.mv = 0
    o.mv = o.m(reset=True)
  def y(o, x, reset=False):
    if   o.phase == 'Intro':
      return o.m(reset)*(x-3)+1
    elif o.phase == 'Removal':
      return o.m(reset)*(x-1)
  def m(o, reset=False):
    if reset:
      if o.phase == 'Intro':
        if   o.category == 'Reqs':
          if   o.sign > 0:
            o.mv = random.uniform(0.0166,.38)
          elif o.sign < 0:
            o.mv = random.uniform(-0.215,-0.035)
        elif o.category == 'Design':
          if   o.sign > 0:
            o.mv = random.uniform(0.0066,0.145)
          elif o.sign < 0:
            o.mv = random.uniform(-0.325,-0.05)
        elif o.category == 'Coding':
          if   o.sign > 0:
            o.mv = random.uniform(0.0066,0.145)
          elif o.sign < 0:
            o.mv = random.uniform(-0.29,-0.05)
      elif o.phase == 'Removal':
        if   o.category == 'Reqs':
          o.mv = random.uniform(0.0,0.14)
        elif o.category == 'Design':
          o.mv = random.uniform(0.0,0.156)
        elif o.category == 'Coding':
          o.mv = random.uniform(0.1,0.176)
    return o.mv


class Sf(Range):
  """Scale Factor"""
  def __init__(o, *args, **kwargs):
    super(o.__class__,o).__init__(*args,**kwargs)
    o.mv = random.uniform(-0.972,-0.648)
    o.calibs = Calibrations(o.txt)
  def y(o,x,reset=False): return o.m(reset)*(x - 6)
  def m(o,reset=False):
    if reset:
      o.mv = random.uniform(-0.972,-0.648)
    return o.mv
  

class Dr(Range):
  """Defect Remover"""
  def __init__(o, *args, **kwargs):
    super(o.__class__,o).__init__(*args,**kwargs)
    o.calibs = Calibrations(o.txt)
  def y(o,x,reset=False):
    pass

class Em(Range):
  """Effort Multiplier"""
  def y(o,x,reset=False): 
    return o.m(reset)*(x-3)+1
  

class Emp(Em):  
  """Effort Multiplier, Positive slope"""
  def __init__(o, *args, **kwargs):
    super(o.__class__,o).__init__(*args,**kwargs)
    o.mv = random.uniform(0.055,0.15)
    o.calibs = Calibrations(o.txt)
  def m(o,reset=False): 
    if reset:
      o.mv = random.uniform(0.055,0.15)
    return o.mv


class Emn(Em):  
  """Effort Multiplier, Negative slope"""
  def __init__(o, *args, **kwargs):
    super(o.__class__,o).__init__(*args,**kwargs)
    o.mv = random.uniform(-0.166,-0.075)
    o.calibs = Calibrations(o.txt)
  def m(o,reset=False): 
    if reset:
      o.mv = random.uniform(-0.166,-0.075)
    return o.mv


class B(Range):
  def __init__(o, *args, **kwargs):
    super(o.__class__,o).__init__(*args,**kwargs)
    o.rval = random.random()
  def y(o,x,reset=False): 
    if reset:
      o.rval = random.random()
    return -0.036 * x + 1.1 - 0.1*o.rval - 0.05

import os
import sys
def coced0(output=os.environ["HOME"]+"/tmp",
           data = "./data",
           model=None):
  if not model:
    if len(sys.argv) > 1:
      model = sys.argv[1]
    else:
      model="flight"
  _c = Cocomo(data + '/' + model)
  _c.xys(verbose=False)
  out = output + "/" + model + ".csv"
  _c.trials(out=out,verbose=False)
  sys.stderr.write("# see" + out + "\n")


#coced0()

def coced1(max=1000):
  import matplotlib.pyplot as plt

  random.seed(1)
  c = Cocomo()
  n = 0
  out= sorted([c.xy() for x in range(max)],
              key=lambda x: x[1])
  xs=[]
  ys=[]
  for x,y in out:
    n += 1
    xs.append(n)
    ys.append(y)
  p1, = plt.plot(xs,ys,'ro')
  p2, = plt.plot(xs,[x*2 for x in ys],'bo')
  plt.legend([p2,p1],["small","bigger"],loc=4)
  plt.xlim(0,1050)
  plt.yscale('log')
  plt.ylabel('effort')
  plt.xlabel('all efforts, sorted')

  plt.show()
  #plt.savefig('coced1.png')

#coced1()

def coced1b(max=1000):
  import matplotlib.pyplot as plt

  random.seed(1)
  c   = Cocomo()
  n   = 0
  out = sorted([c.xy() for x in range(max)],
               key = lambda x: x[1])
  xs = []
  y1s = []
  y2s = []
  y3s = []
  for x,y1,y2,y3 in out:
    n += 1
    xs.append(n)
    y1s.append(y1)
    y2s.append(y2)
    y3s.append(y3)



def coced2(max=1000,rounds=10):
  #random.seed(1)
  c = Cocomo()
  coced2a(rounds,c,max)


def coced2a(r,c,max,updates={}):
  def h100(x,r=250) : return int(x/r) * r
  if r > 0:
    for k in updates:
      c.all[k].sample = updates[k]
    out = [c.xy() for x in range(max)]
    efforts = Rsteps("effort[%s]" % r,final=h100)
    for _,effort in out: 
      efforts.all.append(effort)
    somed0(efforts,n=max)
    better = elite(out)
    #for k,v in better.items():
     # print "\n",k
      #somed0(v,n=max)
    coced2a(r-1,c,max,better)
  


 

def coced3(max=1000,rounds=20):
  random.seed(1)
  c = Cocomo()

  import matplotlib.pyplot as plt

  #plt.yscale('log')
  plt.ylabel('effort')
  plt.xlabel('all efforts, sorted')
  styles=["r-","m-","c-","y-","k-","b-","g-"]  
  plots=[]
  legends=[]
  coced3a(0,len(styles)-1,c,max,plt,styles,plots=plots,legends=legends)
  plt.legend(plots,legends,loc=2)
  plt.xlim(0,1050)
  plt.show()


def coced3a(round,rounds,c,max,plt,styles,updates={},plots=[],legends=[]):
  def h100(x,r=250) : return int(x/r) * r
  if round <= rounds:
    for k in updates:
      c.all[k].sample = updates[k]
    out = [c.xy() for x in range(max)]
    better = elite(out)
    plot = plt.plot([x for x in range(1000)],
                    sorted([effort for _,effort in out]),
                    styles[round],linewidth=round+1)
    plots.append(plot)
    legends.append("round%s" % round)

    coced3a(round+1,rounds,c,max,plt,styles,updates=better,
            plots=plots,legends=legends)


def coced4(samples=1000,rounds=15):
  #random.seed(1)
  c = Cocomo()
  import matplotlib.pyplot as plt
  #plt.yscale('log')
  xs = []
  medians=[]
  spreads=[]
  mosts=[]
  coced4a(0,rounds,c,samples,{},xs,medians,spreads,mosts)
  plt.ylabel('effort')
  plt.xlabel('round')
  plt.legend([plt.plot(xs,medians),plt.plot(xs,spreads)],
              ["median","spread"],
              loc=1)
  plt.xlim(-0.5,len(medians)+0.5)
  plt.ylim(0,1.05*max(medians + spreads + mosts))
  plt.show()


def coced4a(round,rounds,c,samples,updates={},xs=[],medians=[],spreads=[],mosts=[]):
  if round <= rounds:
    print round
    for k in updates:
      if not c.all[k].wild:
        c.all[k].sample = updates[k]
        somed0(c.all[k].sample,n=100)
    out = [c.xy() for x in range(samples)]
    better = elite(out)
    ys = sorted([x for _,x in out])
    p25,p50,p75= [int(len(ys)*n) for n in [0.25,0.5,0.75]]
    medians.append(ys[p50])
    spreads.append(ys[p75] - ys[p25])
    xs.append(round)
    coced4a(round+1,rounds,c,samples,updates=better,
            xs=xs,medians=medians,spreads=spreads,mosts=mosts)  


def elite(xy,bins=7,top=0.2,final=float,key=lambda x:x[1]):
  def r(x) : return "%3.2f" % x
  def keep(lst):
    keeper = {}
    for how,_ in lst:
      if not keeper:
        for k in how: 
          keeper[k] = Rsteps(k,bins,final)
      for k,v in how.items():
        keeper[k].put(v)
    return keeper
  n     = int(top*len(xy))
  xy    = sorted(xy,key=key)
  bests = keep(xy[:n])
  rests = keep(xy[n:])
  for k,v in bests.items():
    print k, bests[k] - rests[k]
  return bests


#coced4()


