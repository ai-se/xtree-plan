from __future__ import division, print_function


class cliffs():

  def __init__(self, lst1, lst2,
               dull=[0.147,  # small
                     0.33,  # medium
                     0.474  # large
                     ]):
    self.lst1, self.lst2 = lst1, lst2
    self.dull = dull

  def delta(self):
    m, n = len(self.lst1), len(self.lst2)
    dom = lambda a, b: -1 if a < b else 1 if a > b else 0
    dominationMtx = [[dom(a, b) for a in self.lst1] for b in self.lst2]
    delta = sum([sum(b) for b in dominationMtx]) / (m * n)
    return delta

  def comment(self):
    cd = self.delta()
    if abs(cd) < self.dull[0]:
      return "Negligible"
    if self.dull[0] < abs(cd) < self.dull[1]:
      return "Small"
    if self.dull[1] < abs(cd) < self.dull[2]:
      return "Medium"
    if self.dull[2] < abs(cd):
      return "Large"


def _cliffsDelta():
  from random import randint
  N = 1000
  list1 = [randint(0, 1) for _ in xrange(N)]
  list2 = [randint(0, 1) for _ in xrange(N)]
  print('Testing Cliffs Delta')
  print(21 * '`')
  print("list1: ", list1)
  print("list2: ", list2)
  cd = cliffs(lst1=list1, lst2=list2)
  print('CliffsDelta:', cd.delta(), '\nDifference: ', cd.comment())

if __name__ == '__main__':
  _cliffsDelta()
