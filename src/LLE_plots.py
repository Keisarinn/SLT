import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pyl

from matplotlib import offsetbox
from mpl_toolkits.mplot3d import Axes3D



def plot_embedding_2D(X,labels, metric, ncomponents, nneighbors):
    
    
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    

    fig,ax = plt.subplots(1)

    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(labels[i]),
                 color=plt.cm.Set1(labels[i] / 10.),
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


    plt.xticks([]), plt.yticks([])

    title = '# of Components: ' + str(ncomponents) + ', # of Neighbors: ' + str(nneighbors) + ', Metric used: ' + metric
    suptitle = '2D LLE Plot'


    fig.suptitle(suptitle,fontsize=14,fontweight='bold')
    ax.set_title(title)

    file_name = '2D_LLE_plot_c_' + str(ncomponents) + '_n_' + str(nneighbors) + '_m_' + metric

    plt.savefig(file_name,dpi=2000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None)

    plt.close()

    
    
def plot_embedding_3D(X,labels,metric, ncomponents, nneighbors):
    
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    for i in range(X.shape[0]):
        ax.text(X[i, 0], X[i, 1],X[i,2], str(labels[i]),
                 color=plt.cm.Set1(labels[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([]), plt.yticks([])

    title = '# of Components: ' + str(ncomponents) + ', # of Neighbors: ' + str(nneighbors) + ', Metric used: ' + metric
    suptitle = '3D LLE Plot'

    fig.suptitle(suptitle, fontsize=14, fontweight='bold')
    plt.title(title)

    file_name = '3D_LLE_plot_c_' + str(ncomponents) + '_n_' + str(nneighbors) + '_m_' + metric

    plt.savefig(file_name, dpi=2000, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1,
                frameon=None)

    plt.close()

def matrix_plot(M):

    plt.matshow(M,fignum=100, cmap=plt.cm.gray)
    plt.show()
