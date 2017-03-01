import sklearn.manifold as manifold
import mnist_access as ma
import proj_one_constants as const
import matplotlib.pyplot as plt
import plot_util as util
import matplotlib.pyplot as plt
import LLE_implementation as LLE

# Load mnist dataset
test_images, test_labels = ma.get_labeled_data(const.TEST_IM, const.TEST_LABELS) # First test set as it is smaller and quicker to load
# train_set = ma.get_labeled_data(const.TRAIN_IM, const.TRAIN_LABELS

#Turn into matrix
test_matrix = ma.to_feature_matrix(test_images)

# Run the LLE on a subset of the whole dataset
RESTRIC_NUM = 2000
NN = 5
test_restrict = test_matrix[0:RESTRIC_NUM,:]
test_labels_restrict = test_labels[0:RESTRIC_NUM, 0]

# lle = LLE.fit_LLE(X=test_restrict, n_neighbours=5, n_components=2)
#
# print "dio"
#
# W = LLE.find_weights(test_restrict, n_neighbours=5, norm=2)
# M = LLE.get_M(W)

lle_2D = manifold.LocallyLinearEmbedding(n_neighbors=NN, n_components=2)
test_2D = lle_2D.fit_transform(test_restrict)

print params
