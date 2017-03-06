import idx2numpy
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import cm
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import Axes3D


def flatten(images):
    flattened = np.empty(shape=(images.shape[0], images.shape[1] * images.shape[2]))
    for i, image in enumerate(images):
        flattened[i] = image.flatten()
    return flattened


def locally_linear_embedding(X, n_neighbors, n_components, distance_metric='minkowski'):
    # find nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=distance_metric).fit(X)
    _, indices = nbrs.kneighbors(X)
    indices = indices[:, 1:]  # removes first because identical to point
    # compute W
    W = np.zeros(shape=(len(X), len(X)))
    for i, x in enumerate(X):
        # make matrix C
        C = np.empty(shape=(n_neighbors, n_neighbors))
        for row_nb, j in enumerate(indices[i]):
            for col_nb, k in enumerate(indices[i]):
                C[row_nb][col_nb] = np.dot(x - X[j], x - X[k])
        C_inv = np.linalg.inv(C)
        for row_nb, j in enumerate(indices[i]):
            W[i][j] = np.sum(C_inv, axis=1)[row_nb] / np.sum(C_inv)
    # compute M
    M = np.dot(np.transpose(np.identity(len(X)) - W), np.identity(len(X)) - W)
    # extract Y
    w, v = np.linalg.eig(M)
    idx = w.argsort()[::-1]
    v = v[:, idx]
    Y = v[:, -n_components - 1:-1]
    return Y, M


def plot_matrix():
    ndarr1 = idx2numpy.convert_from_file("./data/train-images.idx3-ubyte")
    X = flatten(ndarr1[0:100])
    _, M = locally_linear_embedding(X, 5, 2, distance_metric='euclidean')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(M)
    fig.colorbar(cax)
    fig.suptitle('Matrix plot of M with 100 data points and 5 neighbours (Euclidean distance)')
    plt.show()


def plot_linear_interpolation():
    ndarr1 = idx2numpy.convert_from_file("./data/train-images.idx3-ubyte")

    fig = plt.figure()
    fig.suptitle('Linear interpolation between two images (labels 3 and 1)')
    for i in range(0, 8):
        trans = ndarr1[8] * i / 7 + ndarr1[98] * (7 - i) / 7
        plt.subplot(2, 4, i + 1)
        plt.imshow(trans, cmap='gray')
    plt.show()


def plot2D():
    ndarr1 = idx2numpy.convert_from_file("./data/train-images.idx3-ubyte")
    ndarr2 = idx2numpy.convert_from_file("./data/train-labels.idx1-ubyte")
    n = 1000
    X = flatten(ndarr1[0:n])
    labels = ndarr2[0:n]

    y, _ = locally_linear_embedding(X, 5, 2, distance_metric='euclidean')
    fig = plt.figure()
    fig.suptitle('Scatter plot of 1000 data points in a 2D embedding space (5 neighbours, Euclidean distance)')
    ax = fig.add_subplot(111)
    for i in range(0, n):
        ax.scatter(y[i, 0], y[i, 1], marker=r"$ {} $".format(str(labels[i])), c=cm.rainbow(labels[i] * 0.1),
                   edgecolors='none', s=70)
    plt.show()


def plot3D():
    ndarr1 = idx2numpy.convert_from_file("./data/train-images.idx3-ubyte")
    ndarr2 = idx2numpy.convert_from_file("./data/train-labels.idx1-ubyte")
    n = 1000
    X = flatten(ndarr1[0:n])
    labels = ndarr2[0:n]

    y, _ = locally_linear_embedding(X, 5, 3, distance_metric='euclidean')
    fig = plt.figure()
    fig.suptitle('Scatter plot of 1000 data points in a 3D embedding space (5 neighbours, Euclidean distance)')
    ax = fig.add_subplot(111, projection='3d')
    for i in range(0, n):
        ax.view_init(20, -140)
        ax.scatter(y[i, 0], y[i, 1], y[i, 2], marker=r"$ {} $".format(str(labels[i])), c=cm.rainbow(labels[i] * 0.1),
                   edgecolors='none', s=70)
    plt.show()
