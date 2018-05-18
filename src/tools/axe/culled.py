Class NumL
  def winLoss(i,j,reverse=False, 
              same=lambda a,b,c:ttest(a,b,c),
              conf=The.math.brink.conf):
    if same(i,j,conf=conf):
      i.tie += 1; j.tie += 1
      return False # no one win or lost
    iBest= i.mu < j.mu if reverse else i.mu > j.mu
    if iBest:
      i.win += 1; j.loss+= 1
    else:
      i.loss+= 1; j.win += 1 
    return True # we have a winner
