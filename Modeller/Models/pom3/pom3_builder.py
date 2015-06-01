import sys,re
import pom3


header = []
rows = []

def pom3_csvmaker(names,rows,verbose=True):
    header = names[:]
    objectives = ['-cost','+completion','-idle']
    p3 = pom3.pom3()
    bigrows = []
    #clean rows
    for _r,r in enumerate(rows):
        for _i,i in enumerate(r):
            rows[_r][_i] = round(float(i),2)
    #simulate pom3
    for r in rows:
        vals = p3.simulate(r)
        bigrow = r+[round(i,2) for i in vals]
        bigrows.append(bigrow)
    
    header += objectives
    
    if verbose:
        s = ''
        for i in header:
            s += str(i)+','
        print s[:len(s)-1]
        for r in bigrows:
            s = ''
            for i in r:
                s +=str(i) + ','
            print s[:len(s)-1]
    
    return header,bigrows
