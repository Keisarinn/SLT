import os
from mnist import MNIST
import numpy as np
from numpy import unravel_index


#Set up paths
__file__ = "/data/train-images.idx3-ubyte"
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
curr_cwd = os.getcwd()
# train_file = "data/train-images.idx3-ubyte"
# __location__ = os.path.join(curr_cwd, train_file)

mndata = MNIST(os.path.join(curr_cwd,"data"))
images, labels = mndata.load_training()
images = np.asarray(images[:200])
labels = np.asarray(labels[:200])

import matplotlib.pyplot as plt

# This import is needed to modify the way figure behaves
from mpl_toolkits.mplot3d import Axes3D
Axes3D
from sklearn import manifold

def LLE_sk(components = 2, neighbors = 4):
    X_r, err = manifold.locally_linear_embedding(images, n_neighbors=neighbors,
                                                    n_components=components)

    ax = plt.subplot(212, projection='3d')
    ax.scatter(X_r[:, 0], X_r[:, 1], X_r[:, 2], c=labels, cmap=plt.cm.Spectral)
    ax.set_aspect('auto')
    plt.axis('tight')
    # plt.xticks([]), plt.yticks([])
    plt.title('LLE 3D')
    ax = plt.subplot(211)
    ax.scatter(X_r[:, 0], X_r[:, 1], c=labels, cmap=plt.cm.Spectral)
    ax.set_title("LLE 2D")
    ax.set_aspect('auto')
    plt.show()


def LLE(k = 4):

    n = len(images)

    distances = np.zeros((n,n))

    # 1. Calculate distances and choose k nearest neighbors
    for i in range(n):
        point = images[i]
        for j in range(n):
            other_point = images[j]
            diff = abs(point - other_point)
            distance = np.sum(diff)
            distances[i,j] = distance

    # Select k closest neighbors
    distances_min = np.zeros((n,k))
    distances_min_index = np.zeros((n,k))
    for i in range(n):
        it = 0
        for j in [x for x in range(n) if x != i]:
            if it < k:
                distances_min[i][it] = abs(distances[i][j])
                distances_min_index[i][it] = j
                it+=1
            else:
                max_distance = np.max(distances_min)
                if distances[i][j] < max_distance:
                    max_index = unravel_index(distances_min[i].argmax(),distances_min[i].shape)
                    distances_min_index[i][max_index] = j #Update index for distance
                    new_dist = distances[i][j]
                    distances_min[i][max_index] = new_dist # Change the distance


    # We now have nearest neighbors for every point in distances_min_index

    #Build up the Z matrix
    Z_index =   [[int(j) for j in x ]for x in distances_min_index]
    Z = [[[0]*images.shape[1] for x in range(k)] for i in range(n)]
    for i in range(n):
        it = 0
        for j in Z_index[i]:
            Z[i][it] = images[j]
            it+=1
        Z[i] = Z[i] - images[i]

    # Should be centered around origo

    a = 2

    # 2. Solve for the reconstruction weights W










LLE(4)