def atom(x):
  try : return int(x)
  except ValueError:
    try : return float(x)
    except ValueError : return x
