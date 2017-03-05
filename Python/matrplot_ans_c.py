import sklearn.manifold as manifold
import mnist_access as ma
import proj_one_constants as const
import matplotlib.pyplot as plt
import plot_util as util
import matplotlib.pyplot as plt
import LLE_implementation as LLE
import sklearn_lle_implement as sklle
import numpy as np
import scipy as sp

# Load mnist dataset
test_images, test_labels = ma.get_labeled_data(const.TEST_IM, const.TEST_LABELS) # First test set as it is smaller and quicker to load
# train_set = ma.get_labeled_data(const.TRAIN_IM, const.TRAIN_LABELS

#Turn into matrix
test_matrix = ma.to_feature_matrix(test_images)

# Run the LLE on a subset of the whole dataset
RESTRIC_NUM = 10000
indices = np.random.randint(test_images.shape[0], size=RESTRIC_NUM)
NN = 5
test_restrict = test_matrix[indices,:]
test_labels_restrict = test_labels[indices, 0]

# lle = LLE.fit_LLE(X=test_restrict, n_neighbours=5, n_components=2)
#
# print "dio"
#
# W = LLE.find_weights(test_restrict, n_neighbours=5, norm=2)
# M = LLE.get_M(W)

lle_2D, M = sklle.locally_linear_embedding(test_restrict, n_neighbors=NN, n_components=2)

M_mat = sp.sparse.csr_matrix(M)
M_dens = M_mat.todense()

plt.imshow(M_dens, cmap=plt.cm.ocean, interpolation='none')
plt.colorbar()
# plt.spy(M_mat, marker=None, markersize=None, cmap=plt.cm.ocean)

# svd
print ">>SVD<<"
plt.figure()
U, s, Vh = sp.linalg.svd(M_dens)
print "end SVD"
sort_s = np.matrix.sort(s)
print "end sort"
plt.plot(s, '*-')


plt.show()




# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# ax.set_aspect('equal')
# plt.imshow(M_mat, interpolation='nearest', cmap=plt.cm.ocean)
# plt.colorbar()
# plt.show()
