# Use the sklearn implementation of LLE to explore the dataset
import sklearn.manifold as manifold
import mnist_access as ma
import proj_one_constants as const
import matplotlib.pyplot as plt

from matplotlib import offsetbox
import numpy as np

from mpl_toolkits.mplot3d import Axes3D

import giac_lle as myLLE

import sklearn_lle_implement as implement

# ------------------------------------------------------------
def plot_embedding(X, y, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.TextArea(y[i]),
                X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

def plot_3D_embedding(X, y, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111, projection='3d')
    for i in range(X.shape[0]):
        ax.text(X[i, 0], X[i, 1],X[i,2], str(y[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    # if hasattr(offsetbox, 'AnnotationBbox'):
    #     # only print thumbnails with matplotlib > 1.0
    #     shown_images = np.array([[1., 1.]])  # just something big
    #     for i in range(X.shape[0]):
    #         dist = np.sum((X[i] - shown_images) ** 2, 1)
    #         if np.min(dist) < 4e-3:
    #             # don't show points that are too close
    #             continue
    #         shown_images = np.r_[shown_images, [X[i]]]
    #         imagebox = offsetbox.AnnotationBbox(
    #             offsetbox.TextArea(y[i]),
    #             X[i])
    #         ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


# -----------------------------------------------------------

# Load data
# First test set as it is smaller and quicker to load
test_images, test_labels = ma.get_labeled_data(const.TEST_IM, const.TEST_LABELS)
# train_set = ma.get_labeled_data(const.TRAIN_IM, const.TRAIN_LABELS

print "test labels shape: " + str(test_labels.shape)
#Turn into matrix
test_matrix = ma.to_feature_matrix(test_images)
# Restrict matrix so for testing purposes
RESTRIC_NUM = 2000
test_restrict = test_matrix[0:RESTRIC_NUM,:]
# print test_restrict.shape

#
# # Setup LLE
# # 5 Neighbours
lle = implement.LocallyLinearEmbedding(n_neighbors=5, n_components=2)
transformed_test = lle.fit_transform(test_restrict)

# plt.scatter(transformed_test[:,0],transformed_test[:,1], cmap=plt.cm.Spectral)
plot_embedding(X=transformed_test, y=test_labels[0:RESTRIC_NUM, 0])

# ------------------------------------ 3D PLOT ---------------------------------
test_restrict = test_matrix[0:RESTRIC_NUM,:]

# Setup LLE
# 5 Neighbours 3D
lle = implement.LocallyLinearEmbedding(n_neighbors=5, n_components=3)
transformed_test = lle.fit_transform(test_restrict)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

plot_3D_embedding(X=transformed_test, y=test_labels[0:RESTRIC_NUM, 0])


# # -------------------------------- 2D PLOT with MyLLE --------------------------
#
# myLLE_result = myLLE.fit_LLE(X=test_restrict, n_neighbours=5, n_components=2)
# plot_embedding(X=myLLE_result, y=test_labels[0:RESTRIC_NUM, 0])
#
# # -------------------------------- 3D PLOT with MyLLE --------------------------
#
# t_restrict = test_matrix[0:RESTRIC_NUM,:]
#
# # Setup LLE
# # 5 Neighbours 3D
# myLLE_result = myLLE.fit_LLE(X=test_restrict, n_neighbours=5, n_components=3)
#
# # fig = plt.figure()
# # ax = fig.add_subplot(111, projection='3d')
#
# plot_3D_embedding(X=myLLE_result, y=test_labels[0:RESTRIC_NUM, 0])

plt.show()
