class run():
  def __init__(self, pred = CART, _smoteit = True, _n = -1,
               _tuneit = False, dataName = None, reps = 10):
    self.pred = pred
    self._smoteit = _smoteit
    self.tunedParams = None if not _tuneit else tuner(p, train[_n])
    self.train = dataName
    self.reps = reps
    self._n = _n
  
  def categorize(self):
    
  def go(self):
    one, two = explore(dir)
    data = [one[i] + two[i] for i in xrange(len(one))];
    train = [dat[0] for dat in withinClass(data[n])]
    test = [dat[1] for dat in withinClass(data[n])]



