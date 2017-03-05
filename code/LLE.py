# imports
from mnist import MNIST

import numpy as np

from matplotlib import pyplot as plt
from matplotlib import offsetbox
from mpl_toolkits.mplot3d import Axes3D
%matplotlib osx

from scipy.linalg import eigh, svd, qr, solve
from scipy.sparse import eye, csr_matrix

from sklearn.neighbors import NearestNeighbors
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)

from time import time
import image_util

# get the data
mndata = MNIST('../data/')

train_images, train_labels = mndata.load_training() # 60000 images of 28x28
test_images , test_labels  = mndata.load_testing()  # 10000 images of 28x28

# convert to numpy arrays
X , Y = np.array(train_images), np.array(train_labels)

test_images  = np.array(test_images)
test_labels  = np.array(test_labels)

n_samples, n_features = X.shape

# use only n_points data points
n_points = 1000
x = X[:n_points,]
y = Y[:n_points,]

# Scale and visualize the embedding vectors
def plot_2d_embedding_subplot(X, Y, k, n_neighbors):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    
    ax = plt.subplot(k)
    ax.set_title("Number of neighbors: {}".format(n_neighbors))
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(Y[i]),
                 color=plt.cm.Set1(Y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([]), plt.yticks([])

def plot_3d_embedding_subplot(X, Y, k, n_neighbors):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    
    ax = plt.subplot(k, projection='3d')
    ax.set_title("Number of neighbors: {}".format(n_neighbors))
    for i in range(X.shape[0]):
        ax.text(X[i, 0], X[i, 1], X[i, 2], str(Y[i]),
                color=plt.cm.Set1(Y[i] / 10.),
                fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([]), plt.yticks([])

def visualize_data(train_sample_number):
    plt.imshow(X[train_sample_number].reshape(28,28), interpolation='nearest')
    plt.show()

def locally_linear_embedding(X, n_neighbors, n_components):
    n_samples, n_features = X.shape
    
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1)
    nbrs.fit(X)
    X = nbrs._fit_X
    
    W = manifold.locally_linear.barycenter_kneighbors_graph(X, n_neighbors).toarray()
    
    M = np.eye(n_samples) - W
    M = np.dot(M.T, M)
    
    e_values, e_vectors =  eigh(M, eigvals=(1, n_components))
    e_vectors = e_vectors[:, e_values.argsort()[::-1]]
    
    return e_vectors

