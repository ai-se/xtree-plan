from pandas import DataFrame, read_csv, concat
from os import walk
import numpy as np
from pdb import set_trace
import sys

def say(text):
  sys.stdout.write(str(text))

def shuffle(df, n=1, axis=0):     
    df = df.copy()
    for _ in range(n):
        df.apply(np.random.shuffle, axis=axis)
    return df
    

def csv2DF(dir, as_mtx=False, toBin=False):
  files=[]
  for f in dir:
    df=read_csv(f)
    headers = [h for h in df.columns if '?' not in h]
    # set_trace()
    if isinstance(df[df.columns[-1]][0], str):
      df[df.columns[-1]] = DataFrame([0 if 'N' in d or 'n' in d else 1 for d in df[df.columns[-1]]])
    if toBin:
      df[df.columns[-1]]=DataFrame([1 if d > 0 else 0 for d in df[df.columns[-1]]])
    files.append(df[headers])
  "For N files in a project, use 1 to N-1 as train."
  data_DF = concat(files)
  if as_mtx: return data_DF.as_matrix()
  else: return data_DF

def explore(dir='../data.dat/Jureczko/', name=None):
  datasets = []
  for (dirpath, dirnames, filenames) in walk(dir):
    datasets.append(dirpath)
  training = []
  testing = []
  if name:
      for k in datasets[1:]:
        if name in k:
          if 'Jureczko' or 'mccabe' in dir:
            train = [[dirPath, fname] for dirPath, _, fname in walk(k)]
            test = [train[0][0] + '/' + train[0][1].pop(-1)]
            # set_trace()
            training = [train[0][0] + '/' + p for p in train[0][1] if not p == '.DS_Store' and '.csv' in p]
            testing = test
            return training, testing
          elif 'Seigmund' in dir:
            train = [dir+name+'/'+fname[0] for dirPath, _, fname in walk(k)]
            return train
  else:
    for k in datasets[1:]:
      train = [[dirPath, fname] for dirPath, _, fname in walk(k)]
      test = [train[0][0] + '/' + train[0][1].pop(-1)]
      training.append(
          [train[0][0] + '/' + p for p in train[0][1] if not p == '.DS_Store'])
      testing.append(test)
    return training, testing
