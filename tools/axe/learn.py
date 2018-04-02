from __future__ import division
import sys
sys.dont_write_bytecode = True
from sklearn.ensemble import RandomForestClassifier 
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from abcd import *

def trainTest(tests,trains,indep,dep):
  x1=[]; y1=[]
  x2=[]; y2=[]
  for train in trains:
    x1 += [indep(train)]
    y1 += [dep(train)]
  for test in tests:
    x2 += [indep(test)]
    y2 += [dep(test)]
  return x1,y1,x2,y2

def learns(tests,trains,indep=lambda x: x[:-1],
                    dep = lambda x: x[-1],
                    rf  = Abcd(),
                    lg  = Abcd(),
                    dt  = Abcd(),
                    nb  = Abcd()):
  x1,y1,x2,y2= trainTest(tests,trains,indep,dep) 
  forest = RandomForestClassifier(n_estimators = 50)  
  forest = forest.fit(x1,y1)
  for n,got in enumerate(forest.predict(x2)):
    rf(predicted = got, actual = y2[n])
  logreg = linear_model.LogisticRegression(C=1e5)
  logreg.fit(x1, y1)
  for n,got in enumerate(logreg.predict(x2)):
    lg(predicted = got, actual = y2[n])
  bayes =  GaussianNB()
  bayes.fit(x1,y1)
  for n,got in enumerate(bayes.predict(x2)):
    nb(predicted = got, actual = y2[n])
  dectree = DecisionTreeClassifier(criterion="entropy",
                         random_state=1)
  dectree.fit(x1,y1)
  for n,got in enumerate(dectree.predict(x2)):
    dt(predicted = got, actual = y2[n])
