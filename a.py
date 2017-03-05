import numpy as np
import matplotlib.pyplot as plt
import os.path
from time import time
import copy
from mnist_data import *
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import manifold

#http://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/
def vectorize_MNIST(X_train, X_test):
    num_pixels = X_train.shape[2] * X_train.shape[3]
    X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')
    # normalize inputs from 0-255 to 0-1
    #X_train = X_train / 255
    #X_test = X_test / 255
    return X_train, X_test

def plot_embedding(n_neighbors, X, y, component, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
#    plt.figure()
    if component == 2:
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
    elif component == 3:
        ax = plt.subplot(111, projection='3d')
        for i in range(X.shape[0]):
            ax.text(X[i, 0], X[i, 1], X[i,2], str(y[i]),
                     color=plt.cm.Set1(y[i] / 10.),
                     fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    plt.savefig(str(n_neighbors)+'_'+str(component) +'D_'+str(X.shape[0])+'.jpg')

save_path = 'C:/Lasagne/code/SLT_1/'
X_train, y_train, X_test, y_test = load_dataset()
X_train, X_test = vectorize_MNIST(X_train, X_test)
#X_train, X_test = Standardization(X_train, X_test)
n_neighbors = 400
component = 3
number = 3000
print X_test.shape
X_test= X_test[0:number,:]
# Locally linear embedding of the digits dataset
print("Computing LLE embedding")
clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=component, method='standard')

model = clf.fit(X_test)
X_lle = clf.fit_transform(X_test)
plot_embedding(n_neighbors, X_lle, y_test[0:number], component)
print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
M = 0

a = 0

# standardize 5.34775e-05
#not 1.53056e-05