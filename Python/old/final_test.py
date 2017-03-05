import giac_lle as f_LLE
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

# myLLE = f_LLE.fit_LLE(X, n_neighbours=2, n_components=2)

lle = manifold.LocallyLinearEmbedding(n_neighbors=2, n_components=2)
t = lle.fit_transform(X)

lle2 = manifold.locally_linear_embedding(X, 2, 2)

# print 'FINAAAAAAAAAAAAL'
# print myLLE

print 'test'
print t

print lle2
