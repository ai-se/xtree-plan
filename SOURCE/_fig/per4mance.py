#! /Users/rkrsn/miniconda/bin/python
from __future__ import print_function
from __future__ import division
from os import environ
from os import getcwd
from os import walk
from pdb import set_trace
from random import uniform as rand
from random import randint as randi
import pandas
import sys

# Update PYTHONPATH
HOME = environ['HOME']
axe = HOME + '/git/axe/axe/'  # AXE
pystat = HOME + '/git/pystat/'  # PySTAT
cwd = getcwd()  # Current Directory
sys.path.extend([axe, pystat, cwd])

from methods1 import *


def reformat(dir, files):
  """
  Reformat the raw data to suit my other codes.
  **Already done.. DO NOT RUN AGAIN!**
  """
  import csv
  fread = open(dir + files, 'r')
  rows = [line for line in fread]
  header = rows[0].strip().split(',')  # Get the headers
  "Format the headers by prefixing '$' and '<'"
  header = ['$' + h for h in header]
  header[-1] = header[-1][0] + '<' + header[-1][1:]
  body = [[1 if r == 'Y' else 0 if r == 'N' else r for r in row.strip().split(',')]
          for row in rows[1:]]
  "Write Header"
  with open(dir + files, 'w') as fwrite:
    writer = csv.writer(fwrite, delimiter=',')
    writer.writerow(header)
    for b in body:
      writer.writerow(b)

def format tbl
def explorer():
  dir = '../CPM/'
  files = [filenames for (
      dirpath,
      dirnames,
      filenames) in walk(dir)][0]
  for file in files:
    
    
    

def _testMe():
  explorer()

if __name__ == '__main__':
  _testMe()
