import os, sys; from pdb import set_trace
from dectree import *
import pandas as pd 
walk = os.walk

def explore(dir):
 datasets = []
 for (dirpath, dirnames, filenames) in walk(dir):
    datasets.append(dirpath)

 training = []
 testing = []
 for k in datasets[1:]:
  train = [[dirPath, fname] for dirPath, _, fname in walk(k)]
  test = [train[0][0] + '/' + train[0][1].pop(-1)]
  training.append([train[0][0] + '/' + p for p in train[0][1] if not p == '.DS_Store']);
  testing.append(test)
 return training, testing


train, test = explore('./')
data = [train[i]+test[i] for i in xrange(1,len(test))]
template=pd.read_csv(data[0][0], header = 0).columns.get_values().tolist();
for i in data[1:]:
 for k in i:
  tmp = pd.read_csv(k)
  tmp.to_csv(path_or_buf = k ,  header = template, index=False)
set_trace()
pd.to_csv(data[1][0], header= template)
