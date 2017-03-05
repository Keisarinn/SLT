import cPickle, gzip
import numpy as np
import matplotlib.pyplot as plt
import lle as lle
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import axes3d, Axes3D

# Load the dataset
f = gzip.open('data/mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

#Tractor coding. Since we don't want to train a classifier, we
#put all the data in the same tuple
all_images = np.concatenate((train_set[0], valid_set[0], test_set[0]))
all_labels = np.concatenate((train_set[1], valid_set[1], test_set[1]))
all_data = (all_images, all_labels)

#How many data points to include
included = 750
dims = 9
neighbors = 20
metric = 'manhattan'
plot = True


def chi_2_dist(X,Y):
    return sum((X-Y)*(X-Y)/(abs(X+Y)+0.2))


def reconstruct(p, Y, original, n_neighbors, metric):
    #First we find the neighbors of the point in the embedded space
    knn = NearestNeighbors(n_neighbors + 1, metric=metric).fit(Y)
    ind = np.squeeze(knn.kneighbors(p, return_distance=False)[:, 1:])
    neighbors = np.array(Y)[ind]

    w = np.linalg.lstsq(neighbors.T, p)[0]
    if(np.sum(w) != 0):
        w = w/np.sum(w)
    x = np.dot(w.T, original[ind])
    return x

#----------------------------------------------------------------------
def plot_embedding2D(X, title=None):
    #Centering
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()

    #Set color mapping
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(all_labels[i]),
                 color=plt.cm.Set1(all_labels[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    #Plotting
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
#-------------------------------------------------------------------------
def plot_embedding3D(X, title=None):
    # Centering
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for idx, p in enumerate(X_lle):
        ax.scatter(p[0], p[1], p[2], c=plt.cm.Set1(all_labels[idx] / 10.))
    if title is not None:
        plt.title(title)
#-----------------------------------------------------------------------
#Tractor code version of sorting wrt label
sorted_ims = np.array(all_images[1]).reshape(1,784)
for i in range(10):
    for id, im in enumerate(all_images[:included]):
        if all_labels[id] == i:
            sorted_ims = np.vstack([sorted_ims, all_images[id]])
sorted_ims = sorted_ims[1:]


(X_lle, err), M_sparse = lle.locally_linear_embedding(sorted_ims, neighbors, dims, metric=metric)

if plot:
    #Plot data in embedded space
    if dims==2:
        plot_embedding2D(X_lle, 'Projected points, k = 10' )
    else:
        plot_embedding3D(X_lle, 'Projected points, k = 40')
    plt.show()


    #Obtain singular value decomposition
    M = M_sparse.todense()
    U,S,V = np.linalg.svd(M)

    #Plot M-matrix
    plt.imshow(M, cmap='seismic')
    plt.title('M-matrix.')
    plt.show()

    #Plot singular values of M
    plt.plot(range(included), S)
    plt.title('The singular values of M.')
    plt.show()


p1 = 0
p2 = 1

p1_manifold = X_lle[p1]
p2_manifold = X_lle[p2]
steps = [0, 0.3,  0.6,  1]

f, axarr = plt.subplots(4, sharey=True)
for im, i in enumerate(steps):
    p = i*p1_manifold + (1-i)*p2_manifold
    reconstructed_p = reconstruct(p, X_lle, all_images[0:included], neighbors, metric)
    reconstructed_p = reconstructed_p.reshape(28,28)
    axarr[im].imshow(reconstructed_p, cmap='Greys')

f.subplots_adjust(hspace=0)
plt.show()

f, axarr = plt.subplots(4, sharey=True)
for im, i in enumerate(steps):
    original_p = (i*all_images[p1] + (1-i)*all_images[p2]).reshape(28,28)
    axarr[im].imshow(original_p, cmap='Greys')
plt.show()