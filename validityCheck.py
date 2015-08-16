def rowwise_xor(lst):
  return sum(lst) == 1


def isValid(row):
  T1 = row[7] == 1 and row[13] == 1
  T2 = rowwise_xor(row[8:13])
  T3 = rowwise_xor(row[14:18])
  return T1 and T2 and T3
