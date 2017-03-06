from struct import unpack
import gzip
import numpy as np
from numpy import zeros, uint8, float32
from pylab import imshow, show, cm

from time import time
import matplotlib.pyplot as plt
from matplotlib import offsetbox

from sklearn.neighbors import NearestNeighbors

from sklearn import manifold
from mpl_toolkits.mplot3d import Axes3D
Axes3D


def get_labeled_data(imagefile, labelfile):
    """Read input-vector (image) and target class (label, 0-9) and return
       it as list of tuples.
    """
    # Open the images with gzip in read binary mode
    images = gzip.open(imagefile, 'rb')
    labels = gzip.open(labelfile, 'rb')

    # Read the binary data

    # We have to get big endian unsigned int. So we need '>I'

    # Get metadata for images
    images.read(4)  # skip the magic_number
    number_of_images = images.read(4)
    number_of_images = unpack('>I', number_of_images)[0]
    rows = images.read(4)
    rows = unpack('>I', rows)[0]
    cols = images.read(4)
    cols = unpack('>I', cols)[0]

    # Get metadata for labels
    labels.read(4)  # skip the magic_number
    N = labels.read(4)
    N = unpack('>I', N)[0]

    if number_of_images != N:
        raise Exception('number of labels did not match the number of images')

    # Get the data
    x = zeros((N, rows, cols), dtype=float32)  # Initialize numpy array
    y = zeros((N, 1), dtype=uint8)  # Initialize numpy array
    for i in range(N):
        if i % 1000 == 0:
            print("i: %i" % i)
        for row in range(rows):
            for col in range(cols):
                tmp_pixel = images.read(1)  # Just a single byte
                tmp_pixel = unpack('>B', tmp_pixel)[0]
                x[i][row][col] = tmp_pixel
        tmp_label = labels.read(1)
        y[i] = unpack('>B', tmp_label)[0]
    return (x, y)

def view_image(image, label=""):
    print("Label: %s" % label)
    imshow(image, cmap=cm.gray)
    show()

def lle(X, k):
    # OUT OF MEMORY...

    # Neighbors
    neigh = NearestNeighbors(n_neighbors=k+1, metric='minkowski', p=2)
    print X.shape
    neigh.fit(X)
    KN = neigh.kneighbors(X, k+1, return_distance=False)[:,1:]
    print KN.shape

    # Find optimal W
    Z = X[KN] - np.reshape(X.repeat(k, axis=0), (X.shape[0], k, X.shape[1]))
    Zt = np.transpose(Z, axes=(0,2,1))
    C = np.matmul(Zt, Z)
    

if __name__ == '__main__':

    # test_X.shape  = (10000, 28, 28)
    # test_y. shape = (10000, 1)
    # train_X.shape = (60000, 28, 28)
    # train_y.shape = (60000, 1)

    test_X, test_y = get_labeled_data('./data/t10k-images-idx3-ubyte.gz',
                               './data/t10k-labels-idx1-ubyte.gz')

    # train_X, train_y = get_labeled_data('./data/train-images-idx3-ubyte.gz',
                      #          './data/train-labels-idx1-ubyte.gz')


    test_X_flat = np.reshape(test_X, (test_X.shape[0],
		test_X.shape[1] * test_X.shape [2]))

    # lle(test_X_flat, 5)

    print("Computing 3D LLE embedding")
    X_r_3d, err_3d = manifold.locally_linear_embedding(test_X_flat, n_neighbors=3,
                                             n_components=3, n_jobs=-1)
    print("Done. Reconstruction error: %g" % err_3d)

    print("Computing 2D LLE embedding")
    X_r_2d, err_2d = manifold.locally_linear_embedding(test_X_flat, n_neighbors=3,
                                             n_components=2, n_jobs=-1)
    print("Done. Reconstruction error: %g" % err_2d)

    fig = plt.figure()
    ax = fig.add_subplot(211, projection='3d')
    ax.scatter(X_r_3d[:, 0], X_r_3d[:, 1], X_r_3d[:, 2])
    ax.set_title("3D embedding")
    ax = fig.add_subplot(212)
    ax.scatter(X_r_2d[:, 0], X_r_2d[:, 1])
    plt.axis('tight')
    plt.xticks([]), plt.yticks([])
    plt.title('2D embedding')
    plt.show()























