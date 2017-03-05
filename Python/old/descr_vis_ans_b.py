import sklearn.manifold as manifold
import mnist_access as ma
import proj_one_constants as const
import matplotlib.pyplot as plt
import plot_util as util
import matplotlib.pyplot as plt

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

# Fit LLE 2D
lle_2D = manifold.LocallyLinearEmbedding(n_neighbors=NN, n_components=2)
test_2D = lle_2D.fit_transform(test_restrict)

# Fit LLE 3D
lle_3D = manifold.LocallyLinearEmbedding(n_neighbors=NN, n_components=3)
test_3D = lle_3D.fit_transform(test_restrict)

# Plots ------------------------------------------------------------------------
util.plot_2D_embedding(X=test_2D, y=test_labels_restrict)
util.plot_3D_embedding(X=test_3D, y=test_labels_restrict)

plt.show()
