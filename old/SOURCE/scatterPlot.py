from sklearn.decomposition import PCA
from numpy import array
from pdb import set_trace
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages


class scatterPlot():

  def __init__(self, train, test, delta):
    self.train = train._rows
    self.test = test
    self.delta = delta

  def pcaProj(self):
    pca = PCA(n_components=2)
    aa = pca.fit_transform(array([r.cells[:-2] for r in self.train]))
    bb = pca.fit_transform(array([r for r in self.test]))
    cc = pca.fit_transform(array([r.cells[:-2] for r in self.delta]))
    self.scatterplot(
        [aa, bb, cc], c=[[0.5, 0.5, 0.5], [0.85, 0.0, 0.0], [0, 0.85, 0]])

  def scatterplot(self, arr, c):
    whatis = ['Training', 'Testing', 'Planned']
#     pp = PdfPages('fig10.pdf')
#     fig, ax = plt.subplots()
    for i in xrange(len(arr)):
      #       x, y = [x.tolist()[0] for x in arr[i] if -
      #               600 < x.tolist()[0] < 1500 and -
      #               1000 < x.tolist()[1] < 1000], [y.tolist()[1] for y in arr[i] if -
      #                                              600 < y.tolist()[0] < 1500 and -
      # 1000 < y.tolist()[1] < 1000]
      #       x, y = [x.tolist()[0] for x in arr[i]], [y.tolist()[1] for y in arr[i]]

      with open('/Users/rkrsn/git/Transfer-Learning/SOURCE/fig10/' + (whatis[i]), 'a') as fwrite:
        for array in arr[i]:
          fwrite.write('%0.2f,%0.2f\n' % (array[0], array[1]))
#       ax.scatter(x, y, c=c[i], marker='d')
#     pp.savefig()
#     pp.close()
