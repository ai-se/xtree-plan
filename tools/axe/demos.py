from __future__ import division
import sys
sys.dont_write_bytecode = True

def atom(x):
  try : return int(x)
  except ValueError:
    try : return float(x)
    except ValueError : return x

def cmd(com="demo('-h')"):
  "Convert command line to a function call."
  if len(sys.argv) < 2: return com
  def strp(x): return isinstance(x,basestring)
  def wrap(x): return "'%s'"%x if strp(x) else str(x)
  words = map(wrap,map(atom,sys.argv[2:]))
  return sys.argv[1] + '(' + ','.join(words) + ')'

def demo(f=None,cache=[]):
  def doc(d):
    return '# '+d.__doc__ if d.__doc__ else ""
  if f == '-h':
    print '# sample demos'
    for n,d in enumerate(cache):
      print '%3s) ' %(n+1),d.func_name,doc(d)
  elif f:
    cache.append(f);
  else:
    s='|'+'='*40 +'\n'
    for d in cache:
      print '\n==|',d.func_name,s,doc(d),d()
  return f

def test(f=None,cache=[]):
  if f:
    cache += [f]
    return f
  ok = no = 0
  for t in cache:
    print '#',t.func_name ,t.__doc__ or ''
    prefix, n, found = None, 0, t() or []
    while found:
      this, that = found.pop(0), found.pop(0)
      if this == that:
        ok, n, prefix = ok+1, n+1,'CORRECT:'
      else:
        no, n, prefix = no+1, n+1,'WRONG  :'
      print prefix,t.func_name,'test',n
  if ok+no:
    print '\n# Final score: %s/%s = %s%% CORRECT' \
        % (ok,(ok+no),int(100*ok/(ok+no)))

@test
def tested():
  return [True,True,  # should pass
          False,True, # should fail
          1, 2/2]     # should pass

@demo
def demoed(show=1):
  "Sample demo."
  print show/2

@demo
def tests():
  "Run all the test cases."
  test()
