from _model import *
import sys, pdb
import os
_HOME=os.environ["HOME"]
sys.path.insert(0,_HOME+"/git/ai-se/Rahul/DEADANT")
from deadant import *
import math, numpy as np

class Model:
 def __init__(self,name):
  self.name = name
  if name == '_POM3':
   self.model = Pom()
  elif name == 'xomo':
   self.model = Xomo(model = 'flight')
  elif name == 'xomoflight':
   self.model = Xomo(model='flight')
  elif name == 'xomoground':
   self.model = Xomo(model='ground')
  elif name == 'xomoosp':
   self.model = Xomo(model='osp')
  elif name == 'xomoosp2':
   self.model = Xomo(model='osp2')
  elif name == 'xomoall':
   self.model = Xomo(model='all')
  else:
   sys.stderr.write("Enter valid model name _POM3 or xomoflight --> xomo[flight/ground/osp/osp2/all]\n")
   sys.exit()

 def trials(self,N,verbose=False):
  #returns headers and rows
  return self.model.trials(N,verbose)

 def oo(self, verbose=False):
  return self.model.c

 def update(self,fea,cond,thresh):
  #cond is true when <=
  self.model.update(fea,cond,thresh)

 def __repr__(self):
  return self.name

if __name__ == '__main__':
 getModels()
