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


class type1():

  def __init__(self):
    pass

  def striplines(self, line):
    lists = []
    listedline = line[1:-1].strip().split(',')  # split around the = sign
    self.projName = listedline[0][1:-1].strip().split('_')[0]
    lists.append(listedline[0][1:-1])
    for ll in listedline[1:]:
      lists.append(float(ll))
    return lists

  def printHeader(self):
    print(r"""\documentclass{article}
    \usepackage{colortbl}
    \usepackage{fullpage}
    \usepackage{booktabs}
    \usepackage{bigstrut}
    \usepackage[table]{xcolor}
    \usepackage{picture}
    \newcommand{\quart}[4]{\begin{picture}(100,6)
    {\color{black}\put(#3,3){\circle*{4}}\put(#1,3){\line(1,0){#2}}}\end{picture}}
    \begin{document}
    """)

  def printFooter(self):
    print(r"""
    \end{document}
    """)

  def list2sk(self, lst):
    print("\subsection*{" + self.projName + "}")
    rdivDemo(lst, isLatex=True)

  def log2list(self):
    dir = './log'
    self.printHeader()
    files = [filenames for (dirpath, dirnames, filenames) in walk(dir)][0]
    for file in files:
      lst = []
      f = open(dir + '/' + file, 'r')
      for line in f:
        if line:
          lst.append(self.striplines(line[:-1]))
      if lst:
        self.list2sk(lst)
    self.printFooter()


class type2():

  def __init__(self):
    pass

  def striplines(self, line):
    lists = []
    listedline = line[1:-1].strip().split(',')  # split around the = sign
    lists.append(listedline[0][1:-1])
    for ll in listedline[1:]:
      lists.append(float(ll))
    return lists

  def printHeader(self):
    print(r"""\documentclass{article}
    \usepackage{colortbl}
    \usepackage{fullpage}
    \usepackage[table]{xcolor}
    \usepackage{picture}
    \newcommand{\quart}[4]{\begin{picture}(100,6)
    {\color{black}\put(#3,3){\circle*{4}}\put(#1,3){\line(1,0){#2}}}\end{picture}}
    \begin{document}
    """)

  def printFooter(self):
    print(r"""
    \end{document}
    """)

  def list2sk(self, lst):
    print("\section*{}")
    rdivDemo(lst, isLatex=True)

  def log2list(self):
    dir = './log'
    self.printHeader()
    files = [filenames for (
        dirpath,
        dirnames,
        filenames) in walk(dir)][1:]
    dirs = [dirpath for (
        dirpath,
        dirnames,
        filenames) in walk(dir)][1:]
#     set_trace()
    for project, folder in zip(files, dirs):
      lst = []
#       set_trace()
      for file in project:
        f = open(folder + '/' + file, 'r')
        for line in f:
          if line:
            lst.append(self.striplines(line[:-1]))
      if lst:
        self.list2sk(lst)
    self.printFooter()


def _test():
  type1().log2list()

if __name__ == '__main__':
  _test()
