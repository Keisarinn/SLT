import LLE_linear_equations as leq_LLE
import LLE_implementation as LLE_1
import numpy as np
import sklearn.manifold as manifold

X = np.array([[0,0,0],[1,1,1],[0,0,3],[1,2,1]])
# print X.shape
#
# print LLE.nearest_neighbours(X, n_neighbours=2)
#
# Y = np.zeros((4,4))
# ind = np.array([0, 2])
# val = np.array([1413,1123])
# print Y
# Y[1, ind] = val
# print Y

leq = leq_LLE.fit_LLE(X, n_neighbours=2, n_components=2)
lle_1 = LLE_1.fit_LLE(X, n_neighbours=2, n_components=2)

lle = manifold.LocallyLinearEmbedding(n_neighbors=2, n_components=2)
t = lle.fit_transform(X)

print 'LEQ'
print leq
print 'LLE1'
print lle_1
print 'test'
print t
