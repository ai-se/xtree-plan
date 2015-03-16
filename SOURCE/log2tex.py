#! /Users/rkrsn/miniconda/bin/python
from __future__ import print_function, division
from os import environ, getcwd
import sys
from pdb import set_trace

# Update PYTHONPATH
HOME = environ['HOME']
axe = HOME + '/git/axe/axe/'  # AXE
pystat = HOME + '/git/pystats/'  # PySTAT
cwd = getcwd()  # Current Directory
sys.path.extend([axe, pystat, cwd])

from sk import rdivDemo
from os import walk

def striplines(line):
  lists = []
  listedline = line[1:-1].strip().split(',') # split around the = sign
  lists.append(listedline[0][1:-1])
  for ll in listedline[1:]:
    lists.append(float(ll))
  return lists

def list2sk(lst):
  return rdivDemo(lst, isLatex = True)

def log2list():
  lst = []
  dir = './log'
  files = [filenames for (dirpath, dirnames, filenames) in walk(dir)][0]
  for file in files:
    f = open(dir+'/'+file, 'r')
    for line in f:
      lst.append(striplines(line[:-1]))
  print(list2sk(lst))
  
def _test():
  log2list()
  
if __name__ == '__main__':
  _test()
