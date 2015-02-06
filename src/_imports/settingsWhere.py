"""

# Place to store settings.

## Usual Header

"""
import  sys
sys.dont_write_bytecode = True
"""

## Anonymous Containers

"""
class o:
  def __init__(i,**d): i.has().update(**d)
  def has(i): return i.__dict__
  def update(i,**d) : i.has().update(d); return i
  def __repr__(i)   : 
    show=[':%s %s' % (k,i.has()[k]) 
      for k in sorted(i.has().keys() ) 
      if k[0] is not "_"]
    txt = ' '.join(show)
    if len(txt) > 60:
      show=map(lambda x: '\t'+x+'\n',show)
    return '{'+' '.join(show)+'}'

class E:
  def __init__(i,txt):
    i.txt   = txt
    i._f    = None
  def __call__(i,*lst,**d):
    return i.f()(*lst,**d)
  def f(i):
    
    if not i._f: i._f=globals()[i.txt]
    return i._f
  def __repr__(i):
    return i.txt+'()'

def defaults(**d):
  return o(_logo="""
            ,.-""``""-.,
           /  ,:,;;,;,  \ 
           \  ';';;';'  /
            `'---;;---'`
            <>_==""==_<>
            _<<<<<>>>>>_
          .'____\==/____'.
          |__   |__|   __|
         /C  \  |..|  /  D\ 
         \_C_/  |;;|  \_c_/
          |____o|##|o____|
           \ ___|~~|___ /
            '>--------<'
            {==_==_==_=}
            {= -=_=-_==}
            {=_=-}{=-=_}
            {=_==}{-=_=}
            }~~~~""~~~~{
       jgs  }____::____{
           /`    ||    `\ 
           |     ||     |
           |     ||     |
           |     ||     |
           '-----''-----'""",
      what=o(minSize  = 10,    # min leaf size
             depthMin= 2,      # no pruning till this depth
             depthMax= 10,     # max tree depth
             wriggle = 0.2,    # min difference of 'better'
             prune   = True,   # pruning enabled?
             b4      = '|.. ', # indent string
             verbose = False,  # show trace info?
             goal    = lambda m,x : scores(m,x)
             ),
      seed    = 1,
      cache   = o(size=128)
  ).update(**d)

The=None
