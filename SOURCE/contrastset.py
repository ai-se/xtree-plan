import numpy as np

def getContrastSet(loc, myTree):
 contrastSet = {};
 # print loc.f.name, loc.lvl+1, loc.val
 def remember(node):
  key = node.f.name
  Val = node.val
  contrastSet.update({key: Val})
  # print contrastSet
 def forget(key):
  del contrastSet[key]
 def objectiveScores(lst):
  obj = ([k.cells[-2] for k in lst.rows])
  return np.median([k for k in obj]), [k for k in obj]
 def compare(node, test):
   leaves = [n for n in test.kids] if len(test.kids) > 0 else [test]
   for k in leaves:
    return objectiveScores(k) < objectiveScores(node), [objectiveScores(k),
                                                        objectiveScores(node)]
 def trackChanges(testing):
  lvl = testing.lvl
  while lvl > 0:
   lvl = testing.lvl  # @IndentOk
   remember(testing)
   testing = testing.up
 cost = 0
 newNode = loc
 print 'Test Case: '
 print ('Variable name: ', newNode.f.name, 'ID: ', newNode.mode,
        'Value: ', newNode.val, 'Level: ', newNode.lvl + 1)
 print 'No. of Kids: ', len(newNode.kids)
 print 'Cost: ', cost
 def isOnlyNode(node):
  return len(node.kids) <= 1
 while isOnlyNode(newNode):
  # go 1 level up
  newNode = newNode.up;
  # remember(newNode)
  cost += 1
 toScan = [neigh for neigh in newNode.kids if not loc == neigh]
 for testing in toScan:
  isBetter, obj = compare(loc, testing)
  if isBetter:
   trackChanges(testing)
 return contrastSet
