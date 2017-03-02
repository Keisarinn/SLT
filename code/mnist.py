import os
import struct
import numpy as np
import sklearn.manifold as mf
import matplotlib.pyplot as plt
from matplotlib import pyplot
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from sklearn.manifold.locally_linear import barycenter_weights
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state

import animation
from sklearn.neighbors import NearestNeighbors as NN
from numpy import linalg as LA
import sklearn
import scipy


mypath = "./../../../../../../../../Desktop/mnist"
reduce = 3000


def main():
    ''' main method

        To see the plots for the different parts of the coding exercise, just uncomment the right part (part a always needs to be executed)
    '''

    ''' (a) Get data'''
    trainlbl, trainimg = myread(dataset= "training", path= mypath)

    ''' (b) Locally linear embedding '''
    part_b(trainimg, trainlbl)

    ''' (c) Cluster structure '''
    #part_c(trainimg)

    ''' (d) Nearest Neighbors '''
    #part_d(trainimg, trainlbl)

    ''' (e) Linear manifold interpolation '''
    #part_e(trainimg)


def part_b(X, labels):
    for k in [5, 30]:
        for n_components in [2,3]:
            embTrainVectors = applyLLEAdvanced(X[:reduce], False, k, n_components)
            visualizeEmb(embTrainVectors, labels[:reduce], k, n_components, True)

def part_c(X, k =30, n_components = 3):
    embTrainVectors = applyLLEAdvanced(X[:reduce], True, k, n_components)

def part_d(X, labels):
    n_components = 3
    metrics = ['euclidean', 'manhattan']
    for k in [2, 30, 100]:
        for metric in metrics:
            embTrainVectors = applyLLEAdvanced(X[:reduce], False, k, n_components, metric=metric)
            visualizeEmb(embTrainVectors, labels[:reduce], k, n_components, False, metric)


def part_e(X, k = 30, n_components = 3, inManifold = True):
    embTrainVectors = applyLLEAdvanced(X[:reduce], False, k, n_components)

    ''' Reconstruct X '''
    if inManifold:
        # Select random point in manifold
        randind = [303, 305]
        Yi = embTrainVectors[randind,:]
        Xi = X[randind,:]
        X_rec1 = reconstruct(embTrainVectors, Yi[0], k, inManifold, X)
        X_rec2 = reconstruct(embTrainVectors, Yi[1], k, inManifold, X)

        f, ((ax1, ax2, ax3, ax4, ax41), (ax5, ax6, ax7, ax8, ax9)) = plt.subplots(2, 5, sharex='col', sharey='row')

        show(np.reshape(np.array(X_rec1), (28,28)),ax1, sshow=False)      # show reconstructed pic
        ax1.set_xlabel('pi = 0')
        show(linInterpolationOrigSpace(X_rec1, X_rec2, 0.25), ax2, sshow=False)  # show reconstructed pic
        ax2.set_xlabel('pi = 0.25')
        show(linInterpolationOrigSpace(X_rec1, X_rec2, 0.5), ax3, sshow=False)  # show reconstructed pic
        ax3.set_xlabel('pi = 0.5')
        show(linInterpolationOrigSpace(X_rec1, X_rec2, 0.75), ax4, sshow=False)
        ax4.set_xlabel('pi = 0.75')
        show(np.reshape(np.array(X_rec2), (28, 28)), ax41, sshow=False)  # show reconstructed pic
        ax41.set_xlabel('pi = 1')

        show(np.reshape(np.array(Xi[0]), (28,28)),ax5, sshow=False)      # show original pic
        ax5.set_xlabel('pi = 0')
        show(linInterpolationOrigSpace(Xi[0], Xi[1], 0.25),ax6, sshow=False)
        ax6.set_xlabel('pi = 0.25')
        show(linInterpolationOrigSpace(Xi[0], Xi[1], 0.5),ax7, sshow=False)
        ax7.set_xlabel('pi = 0.5')
        show(linInterpolationOrigSpace(Xi[0], Xi[1], 0.75), ax8, sshow=False)
        ax8.set_xlabel('pi = 0.75')
        show(np.reshape(np.array(Xi[1]), (28, 28)), ax9, sshow=False)
        ax9.set_xlabel('pi = 1')

        f.suptitle('Linear Interpolation in Reconstructed and Original Space')
        pyplot.show()


def show(image, ax, sshow = True):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    from matplotlib import pyplot
    import matplotlib as mpl
    if sshow:
        fig = pyplot.figure()
        ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    if sshow:
        pyplot.show()


def linInterpolationOrigSpace(img1, img2, pi):
    '''Each image consists of 28x28 pixels flattened'''
    resultingImg = np.zeros((len(img1),1))
    for i, pixelImg1 in enumerate(img1):
        pixelImg2 = img2[i]
        interpolatedPixelVal = (1-pi) * pixelImg1 + pi * pixelImg2
        resultingImg[i] = interpolatedPixelVal
    resultingImg = np.reshape(np.array(resultingImg), (28,28))
    return resultingImg


def plotSVDMdiag(M):
    u, s, v = LA.svd(M)
    diag = np.diagonal(M)

    val = 28*28
    N = val
    ind = np.arange(N)  # the x locations for the groups
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, s[:N], width, color='r')

    rects2 = ax.bar(ind + width, diag[:N], width, color='b')

    ax.set_title('Singular / Diagonal Values of M')

    plt.show()

def getSingularValues(M):
    u,s,v = LA.svd(M)
    return s

def doMatrixPlot(M):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_aspect('equal')
    plt.imshow(M, cmap=plt.cm.ocean)
    plt.colorbar()
    plt.suptitle('Matrix Plot of M')
    plt.show()


def visualizeEmb(embVec, lbls, k, n_components, makeGif, metric = 'standard'):
    if embVec.shape[1] == 2:
        fig = plt.figure()
        x = embVec[:,0]
        y = embVec[:,1]
        colorarr = np.array(['k', 'g', 'm', 'y', 'c', 'r', 'olive' , 'salmon', 'b', 'orange'])
        colors = colorarr[np.array(lbls)]
        area = np.pi * 2 ** 2
        ax = plt.scatter(x, y, s=area, c=colors, alpha=0.5)
        title = 'Embedded Dimension with k = ' + str(k)
        if not metric == 'standard':
            title += ' and ' + metric + '-Distance-Metric'
        fig.suptitle(title)
        plt.show()
    elif embVec.shape[1] == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = embVec[:, 0]
        y = embVec[:, 1]
        z = embVec[:, 2]
        colorarr = np.array(['k', 'g', 'm', 'y', 'c', 'r', 'olive', 'salmon', 'b', 'orange'])
        colors = colorarr[np.array(lbls)]
        ax.scatter(x,y,z,c=colors)
        title = 'Embedded Dimension with k = ' + str(k)
        if not metric == 'standard':
            title += ' and ' + metric + '-Distance-Metric'

        ax.set_title(title)
        plt.show()

        if makeGif:
            angles = np.linspace(0, 360, 21)[:-1]  # A list of 20 angles between 0 and 360

            # create an animated gif (20ms between frames)
            title = 'nn:' + str(k) + '_space_' + str(n_components) + '.gif'
            animation.rotanimate(ax, angles, title, delay=20)


def myread(dataset = "training", path = "."):

    if dataset is "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError, "dataset must be 'testing' or 'training'"

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows * cols)

    get_img = lambda idx: (lbl[idx], img[idx])

    return lbl, img




"""
LLE Algorithm
"""

def applyLLE(trainimg, k, n_components):
    LLE = mf.LocallyLinearEmbedding(n_neighbors= k, n_components= n_components, neighbors_algorithm= 'kd_tree', method='standard')
    embVectors = LLE.fit_transform(trainimg)

    return embVectors

def applyLLEAdvanced(trainimg, showMPlots, k, n_components, metric = 'minkowski'):
    #W = sklearn.manifold.locally_linear.barycenter_kneighbors_graph(trainimg, k)
    X = trainimg
    n_neighbors = k
    reg = 1e-3
    n_jobs = 1

    knn = NearestNeighbors(n_neighbors + 1, n_jobs=n_jobs, metric=metric).fit(X)
    X = knn._fit_X
    n_samples = X.shape[0]
    ind = knn.kneighbors(X, return_distance=False)[:, 1:]
    data = barycenter_weights(X, X[ind], reg=reg)
    indptr = np.arange(0, n_samples * n_neighbors + 1, n_neighbors)
    W = csr_matrix((data.ravel(), ind.ravel(), indptr),
                      shape=(n_samples, n_samples))

    # we'll compute M = (I-W)'(I-W)
    M = scipy.sparse.eye(*W.shape, format=W.format) - W
    M = (M.T * M).tocsr()

    if hasattr(M, 'toarray'):
        M = M.toarray()
    eigen_values, eigen_vectors = eigh(M, eigvals=(1, n_components), overwrite_a=True)
    index = np.argsort(np.abs(eigen_values))

    if showMPlots:
        '''Do M plots'''
        doMatrixPlot(M)
        plotSVDMdiag(M)

    return eigen_vectors[:, index]


def applyMyLLE(trainimg, k, n_components):

    '''Find Nearest Neigbours'''
    nbrs_obj = sklearn.neighbors.NearestNeighbors(n_neighbors = k + 1, algorithm = 'auto', metric = 'euclidean').fit(trainimg)
    trainimg = nbrs_obj._fit_X
    dist, nbrs = nbrs_obj.kneighbors(trainimg)

    nbrs = nbrs[:,1:]

    print nbrs

    ''' Find M'''
    W = np.zeros((len(trainimg), len(trainimg)))
    for i, nbr in enumerate(nbrs):
        Xi = trainimg[i]    # ith input vector
        Z = np.column_stack((trainimg[nbr] -Xi))    #take all the neighbour vectors and stack them together horizontally; subtract Xi from it
        C = calcLocalCov(Z)     # Compute local covariance
        Wi = LA.solve(C, np.ones((len(C), 1)))
        #Wi = np.dot(np.linalg.inv(C), np.ones((len(C), 1)))     # Solve C*W = 1 for W
        '''if i == 1:
            print Z
            print Wi
            print '----'
            print np.dot(C,Wi)'''
        Wi = Wi / float(np.sum(Wi))   # Scaling
        for idx, j in enumerate(nbr):
            # every row in W has exactly 5 non-zero elements (The ones from its neighbours)
            val = Wi[idx]
            W[i,j] = val
    arg = np.identity(len(W)) - W   #(I-W)
    M = np.dot(np.transpose(arg), arg)  #(I-W)^T (I-W)
    #doMatrixPlot(M)

    svd = getSingularValues(M)
    plotSVDMdiag(M)

    ''' Find Eigenvectors'''
    w, v = LA.eig(M)
    idx = np.argsort(w)[1:n_components + 1]
    botv = v[:,idx] # bottom eigenvectors (bottom one left out)

    '''Set Y'''
    Y = botv
    #print np.array(Y).shape

    return Y

def reconstruct(embSpaceVec,X, n, isInManifold, origSpace):
    '''Find n nearest neighbors for X'''
    if isInManifold == True:
        off = 1
    else:
        off = 0
    nbrs_obj = sklearn.neighbors.NearestNeighbors(n_neighbors= n + off, algorithm='auto', metric='euclidean').fit(embSpaceVec)
    dist, nbrs = nbrs_obj.kneighbors(X)

    nbrs = nbrs[:, off:][0]

    '''Solve for weights'''
    myVecs = np.array(embSpaceVec)[nbrs, :]

    w = LA.lstsq(myVecs.T, X)[0]

    ''' Apply weights to originalSpace'''
    Xorig = np.dot(w.T, origSpace[nbrs, :])

    return Xorig


def calcLocalCov(Z):
    C = np.dot(Z.T, Z)
    if np.trace(C) == 0:
        C += float(1e-3)
    return C



if __name__ == "__main__":
    main()