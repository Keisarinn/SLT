import matplotlib.pyplot as plt
import numpy as np
from matplotlib import offsetbox
from mpl_toolkits.mplot3d import Axes3D


def plot_embedding(X, images, labels, title=None, attach_images=True, pad=.03):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    x_range = x_max - x_min

    plt.figure(figsize=(15,10), dpi=300)
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(labels[i]),
                 color=plt.cm.Set1(labels[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 10})

    if hasattr(offsetbox, 'AnnotationBbox') and attach_images:
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 8e-5:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(images[i], cmap=plt.cm.gray_r),
                X[i], pad=0.)
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    plt.xlim([x_min[0] - pad * x_range[0], x_max[0] + pad * x_range[0]])
    plt.ylim([x_min[1] - pad * x_range[1], x_max[1] + pad * x_range[1]])
    if title is not None:
        plt.title(title, fontsize=16)
    return ax


def plot_embedding_3d(X, labels, title=None):
    x_min, x_max = X.min(0), X.max(0)

    fig = plt.figure(figsize=(15,10), dpi=300)
    ax = fig.add_subplot(111, projection='3d')

    for i in range(X.shape[0]):
        ax.text(X[i,0], X[i,1], X[i,2], str(labels[i]),
        color=plt.cm.Set1(labels[i]/10.),fontdict={'weight':'bold','size':10})

    ax.set_xticks([]), ax.set_yticks([]), ax.set_zticks([])
    ax.set_xlim([x_min[0], x_max[0]]), ax.set_ylim([x_min[1], x_max[1]])
    ax.set_zlim([x_min[2], x_max[2]])
    if title is not None:
        ax.set_title(title, fontsize=16)

def plot_digit(X, truncate=True):
    d = int(np.sqrt(X.size))
    X = X.reshape(d, d)

    plt.figure(figsize=(10,10), dpi=300)
    plt.imshow(X, cmap='gray_r')
