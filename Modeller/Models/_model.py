
#Generic Attribute class to implement in all models
class Attr:
    def __init__(self,name):
        self.name = name
        self.up = 0
        self.low = 0
    
    def update(self,low,up):
        self.low = low
        self.up = up
    
    def __repr__(self):
        s = str(self.low)+' < '+self.name +' < '+str(self.up)+'\n'
        return s

# #POM3 support
# import os,sys,inspect
# cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe()))[0],"_POM3")))
# if cmd_subfolder not in sys.path:
#     sys.path.insert(0, cmd_subfolder)
# 
# from _POM3 import *
# 
# class Pom:
#     def __init__(self):
#         self.collection = {}
#         self.names = ["Culture", "Criticality", "CriticalityModifier", 
#                       "InitialKnown", "InterDependency", "Dynamism", 
#                       "Size", "Plan", "TeamSize"]   
#         LOWS = [0.1, 0.82, 2,  0.40, 1,   1,  0, 0, 1]
#         UPS  = [0.9, 1.20, 10, 0.70, 100, 50, 4, 5, 44]
#         for _n,n in enumerate(self.names):
#             self.collection[n] = Attr(n)
#             self.collection[n].update(LOWS[_n],UPS[_n])
#     
#     def update(self,fea,cond,thresh):
#         ind = self.names.index(fea)
#         if cond:
#             self.collection[fea].update(self.collection[fea].low,
#                                         thresh)
#         else:
#             self.collection[fea].update(thresh,
#                                         self.collection[fea].up)
# 
#     def trials(self,N,verbose=False):
#         inp = []
#         import random
#         for _ in range(N):
#             t = []
#             for n in self.names:
#                 t.append(round(random.uniform(self.collection[n].low,
#                                               self.collection[n].up),2))
#             inp.append(t)
#     
# import _POM3
#         header,rows = pom3_builder.pom3_csvmaker(self.names,inp,verbose)
#         return header,rows

#XOMO support
import os,sys,inspect
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe()))[0],"xomo")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

from xomo import *

class Xomo:
    def __init__(self,
                 out=os.environ["HOME"]+"/git/modeller/xomo",
                 data = "data",
                 model=None):
        def theModel(model):
            #default model is flight
            if not model:
                model = "flight"
            return model
        self.collection = {}
        self.model = theModel(model)
        self.c = Cocomo("xomo/"+data+"/"+self.model)
        self.out = out + "/" + self.model + ".csv"
        self.data = data
        self.names = ["aa", "sced", "cplx", "site", "resl", "acap",
                      "etat", "rely","data", "prec", "pmat", "aexp",
                      "flex", "pcon", "tool", "time","stor", "docu",
                      "b", "plex", "pcap", "kloc", "ltex", "pr", 
                      "ruse", "team", "pvol"] 
        #LOWs and UPs are defined in data/* files according to models
    
        for _n,n in enumerate(self.names):
            self.collection[n] = Attr(n)
            k = filter(lambda x: x.txt == n,self.c.about())[0]
            self.collection[n].update(k.min,k.max)

    def update(self,fea,cond,thresh):
        def applydiffs(c,col,m,thresh,verbose):
            k = filter(lambda x: x.txt == col,c.about())[0]
            if verbose: print k.txt,k.min,k.max,">before"
            if m == "max":
                max = thresh 
                k.update(k.min,max,m=c)
            elif m == "min":
                min = thresh
                k.update(min,k.max,m=c)
            if verbose: print k.txt, k.min, k.max,">after"
        if cond:
            self.collection[fea].up = thresh
            applydiffs(self.c,fea,'max',thresh)
        else:
            self.collection[fea].low = thresh
            applydiffs(self.c,fea,'min',thresh)
    
    def trials(self,N,verbose=False):
        
        for _n,n in enumerate(self.names):
            k = filter(lambda x: x.txt == n,self.c.about())[0]
            if verbose: print k.txt,k.min,k.max,">before"
            k.update(self.collection[n].low,
                     self.collection[n].up,
                     m=self.c)
            if verbose: print k.txt, k.min, k.max,">after"
            if verbose:
                print "Sample of 5"
                for _ in range(5):
                    print n, self.c.xys()[0][n]
                
        
        self.c.xys(verbose=False)
        header,rows = self.c.trials(n=N,out=self.out,verbose=False,write=False)
        return header,rows

def xomod(N=50):
    x = Xomo(model="ground")
    print x.trials(N,False)

# def pom3d(N=50):
#     p = Pom()
#     p.trials(100,True)
    

if __name__ == "__main__":
    #pom3d()
    xomod()
