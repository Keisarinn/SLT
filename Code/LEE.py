import numpy as np
from scipy.linalg import eigh, svd, qr, solve
from scipy.sparse import eye, csr_matrix
# from sklearn import BaseEstimator, TransformerMixin
# from sklearn import check_random_state, check_array
from sklearn.utils.arpack import eigsh
# from sklearn.validation import check_is_fitted
from sklearn.utils.validation import FLOAT_DTYPES
from sklearn.neighbors import NearestNeighbors
import os
from mnist import MNIST
from sklearn import manifold
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
Axes3D
from sklearn.utils import check_array,check_random_state
from numpy import linalg
from scipy.spatial import distance as dist



# Set up paths
__file__ = "/data/train-images.idx3-ubyte"
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
curr_cwd = os.getcwd()
# train_file = "data/train-images.idx3-ubyte"
# __location__ = os.path.join(curr_cwd, train_file)

mndata = MNIST(os.path.join(curr_cwd, "data"))
images, labels = mndata.load_training()
images = np.asarray(images[:500])
labels = np.asarray(labels[:500])

def barycenter_weights(X, Z, reg=1e-3):
    """Compute barycenter weights of X from Y along the first axis
    We estimate the weights to assign to each point in Y[i] to recover
    the point X[i]. The barycenter weights sum to 1.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_dim)
    Z : array-like, shape (n_samples, n_neighbors, n_dim)
    reg: float, optional
        amount of regularization to add for the problem to be
        well-posed in the case of n_neighbors > n_dim
    Returns
    -------
    B : array-like, shape (n_samples, n_neighbors)
    Notes
    -----
    See developers note for more information.
    """
    X = check_array(X, dtype=FLOAT_DTYPES)
    Z = check_array(Z, dtype=FLOAT_DTYPES, allow_nd=True)

    n_samples, n_neighbors = X.shape[0], Z.shape[1]
    B = np.empty((n_samples, n_neighbors), dtype=X.dtype)
    v = np.ones(n_neighbors, dtype=X.dtype)

    # this might raise a LinalgError if G is singular and has trace
    # zero
    for i, A in enumerate(Z.transpose(0, 2, 1)):
        C = A.T - X[i]  # broadcasting
        G = np.dot(C, C.T)
        trace = np.trace(G)
        if trace > 0:
            R = reg * trace
        else:
            R = reg
        G.flat[::Z.shape[1] + 1] += R
        w = solve(G, v, sym_pos=True)
        B[i, :] = w / np.sum(w)
    return B

def barycenter_kneighbors_graph(X,n_neighbors, reg=1e-3, n_jobs=1, metric='minkowski'):
    """Computes the barycenter weighted graph of k-Neighbors for points in X
        Parameters
        ----------
        X : {array-like, sparse matrix, BallTree, KDTree, NearestNeighbors}
            Sample data, shape = (n_samples, n_features), in the form of a
            numpy array, sparse array, precomputed tree, or NearestNeighbors
            object.
        n_neighbors : int
            Number of neighbors for each sample.
        reg : float, optional
            Amount of regularization when solving the least-squares
            problem. Only relevant if mode='barycenter'. If None, use the
            default.
        n_jobs : int, optional (default = 1)
            The number of parallel jobs to run for neighbors search.
            If ``-1``, then the number of jobs is set to the number of CPU cores.
        Returns
        -------
        A : sparse matrix in CSR format, shape = [n_samples, n_samples]
            A[i, j] is assigned the weight of edge that connects i to j.
        See also
        --------
        sklearn.neighbors.kneighbors_graph
        sklearn.neighbors.radius_neighbors_graph
        """
    knn = NearestNeighbors(n_neighbors + 1, n_jobs=n_jobs,metric=metric).fit(X)
    X = knn._fit_X
    n_samples = X.shape[0]
    ind = knn.kneighbors(X, return_distance=False)[:, 1:]
    data = barycenter_weights(X, X[ind], reg=reg)
    indptr = np.arange(0, n_samples * n_neighbors + 1, n_neighbors)
    return csr_matrix((data.ravel(), ind.ravel(), indptr),
                      shape=(n_samples, n_samples))


def null_space(M, k, k_skip=1, eigen_solver='arpack', tol=1E-6, max_iter=100,
               random_state=None):
    """
    Find the null space of a matrix M.
    Parameters
    ----------
    M : {array, matrix, sparse matrix, LinearOperator}
        Input covariance matrix: should be symmetric positive semi-definite
    k : integer
        Number of eigenvalues/vectors to return
    k_skip : integer, optional
        Number of low eigenvalues to skip.
    eigen_solver : string, {'auto', 'arpack', 'dense'}
        auto : algorithm will attempt to choose the best method for input data
        arpack : use arnoldi iteration in shift-invert mode.
                    For this method, M may be a dense matrix, sparse matrix,
                    or general linear operator.
                    Warning: ARPACK can be unstable for some problems.  It is
                    best to try several random seeds in order to check results.
        dense  : use standard dense matrix operations for the eigenvalue
                    decomposition.  For this method, M must be an array
                    or matrix type.  This method should be avoided for
                    large problems.
    tol : float, optional
        Tolerance for 'arpack' method.
        Not used if eigen_solver=='dense'.
    max_iter : maximum number of iterations for 'arpack' method
        not used if eigen_solver=='dense'
    random_state : numpy.RandomState or int, optional
        The generator or seed used to determine the starting vector for arpack
        iterations.  Defaults to numpy.random.
    """
    if eigen_solver == 'auto':
        if M.shape[0] > 200 and k + k_skip < 10:
            eigen_solver = 'arpack'
        else:
            eigen_solver = 'dense'

    if eigen_solver == 'arpack':
        random_state = check_random_state(random_state)
        # initialize with [-1,1] as in ARPACK
        v0 = random_state.uniform(-1, 1, M.shape[0])
        try:
            eigen_values, eigen_vectors = eigsh(M, k + k_skip, sigma=0.0,
                                                tol=tol, maxiter=max_iter,
                                                v0=v0)
        except RuntimeError as msg:
            raise ValueError("Error in determining null-space with ARPACK. "
                             "Error message: '%s'. "
                             "Note that method='arpack' can fail when the "
                             "weight matrix is singular or otherwise "
                             "ill-behaved.  method='dense' is recommended. "
                             "See online documentation for more information."
                             % msg)

        return eigen_vectors[:, k_skip:], np.sum(eigen_values[k_skip:])
    elif eigen_solver == 'dense':
        if hasattr(M, 'toarray'):
            M = M.toarray()
        eigen_values, eigen_vectors = eigh(
            M, eigvals=(k_skip, k + k_skip - 1), overwrite_a=True)
        index = np.argsort(np.abs(eigen_values))
        return eigen_vectors[:, index], np.sum(eigen_values)
    else:
        raise ValueError("Unrecognized eigen_solver '%s'" % eigen_solver)

def null_space(M, k, k_skip=1, eigen_solver='arpack', tol=1E-6, max_iter=100,
               random_state=None):
    """
    Find the null space of a matrix M.
    Parameters
    ----------
    M : {array, matrix, sparse matrix, LinearOperator}
        Input covariance matrix: should be symmetric positive semi-definite
    k : integer
        Number of eigenvalues/vectors to return
    k_skip : integer, optional
        Number of low eigenvalues to skip.
    eigen_solver : string, {'auto', 'arpack', 'dense'}
        auto : algorithm will attempt to choose the best method for input data
        arpack : use arnoldi iteration in shift-invert mode.
                    For this method, M may be a dense matrix, sparse matrix,
                    or general linear operator.
                    Warning: ARPACK can be unstable for some problems.  It is
                    best to try several random seeds in order to check results.
        dense  : use standard dense matrix operations for the eigenvalue
                    decomposition.  For this method, M must be an array
                    or matrix type.  This method should be avoided for
                    large problems.
    tol : float, optional
        Tolerance for 'arpack' method.
        Not used if eigen_solver=='dense'.
    max_iter : maximum number of iterations for 'arpack' method
        not used if eigen_solver=='dense'
    random_state : numpy.RandomState or int, optional
        The generator or seed used to determine the starting vector for arpack
        iterations.  Defaults to numpy.random.
    """
    if eigen_solver == 'auto':
        if M.shape[0] > 200 and k + k_skip < 10:
            eigen_solver = 'arpack'

    if eigen_solver == 'arpack':
        random_state = check_random_state(random_state)
        # initialize with [-1,1] as in ARPACK
        v0 = random_state.uniform(-1, 1, M.shape[0])
        try:
            eigen_values, eigen_vectors = eigsh(M, k + k_skip, sigma=0.0,
                                                tol=tol, maxiter=max_iter,
                                                v0=v0)

        except RuntimeError as msg:
            raise ValueError("Error in determining null-space with ARPACK. "
                             "Error message: '%s'. "
                             "Note that method='arpack' can fail when the "
                             "weight matrix is singular or otherwise "
                             "ill-behaved.  method='dense' is recommended. "
                             "See online documentation for more information."
                             % msg)

        return eigen_vectors[:, k_skip:], np.sum(eigen_values[k_skip:])
    else:
        raise ValueError("Unrecognized eigen_solver '%s'" % eigen_solver)



def main(X, n_neighbors = 4, n_components = 2, reg=1e-3, eigen_solver='auto', tol=1e-6,
        max_iter=100,
        random_state=None, n_jobs=1,metric='minkowski'):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, n_jobs=n_jobs,metric=metric)
    nbrs.fit(X)
    X = nbrs._fit_X
    N, d_in = X.shape
    W = barycenter_kneighbors_graph(
        nbrs, n_neighbors=n_neighbors, reg=reg, n_jobs=n_jobs, metric=metric).todense()
    # M =  eye(*W.shape, format=W.format) - W
    M = np.identity(W.shape[0]) - W
    # M = (M.T * M).tocsr()
    M = (M.T * M)
    return null_space(M, n_components, k_skip=1, eigen_solver=eigen_solver,
                  tol=tol, max_iter=max_iter, random_state=random_state), M




def plot3D(metric = 'minkowski', k = 4):
    components = 3
    X_r, M = main(images, n_neighbors=k, n_components=3, metric = metric)
    X_r = X_r[0]
    # plt.imshow(M)
    # plt.show()

    eig_values, eig_vector = linalg.eig(M)
    eig_values = abs(eig_values)
    idx = np.argsort(eig_values)[1:components + 1]
    botv = eig_vector[:, idx]  # bottom eigenvectors (bottom one left out)
    Y = np.asarray(botv)

    ax = plt.subplot( projection='3d')
    ax.scatter(X_r[:, 0], X_r[:, 1], X_r[:, 2], c=labels, cmap=plt.cm.Spectral)
    ax.set_aspect('auto')
    plt.axis('tight')
    plt.title('LLE 3D: {}'.format(metric))
    # ax = plt.subplot(211, projection='3d')
    # ax.scatter(X_r[:, 0], X_r[:, 1], c=labels, cmap=plt.cm.Spectral)
    # ax.set_title("LLE 2D")
    # ax.scatter(Y[:, 0].real, Y[:, 1].real, Y[:, 2].real, c=labels, cmap=plt.cm.Spectral)
    # ax.set_aspect('auto')
    # plt.title('LLE Y 3D: {}'.format(metric))
    plt.show()

def plot2D(metric = 'minkowski', k = 4):
    components = 2
    X_r, M = main(images, n_neighbors=k, n_components=components, metric=metric)
    X_r = X_r[0]
    # plt.imshow(M)
    # plt.show()

    eig_values, eig_vector = linalg.eig(M)
    eig_values = abs(eig_values)
    idx = np.argsort(eig_values)[1:components + 1]
    botv = eig_vector[:, idx]  # bottom eigenvectors (bottom one left out)
    Y = np.asarray(botv)

    ax = plt.subplot(212)
    ax.scatter(X_r[:, 0], X_r[:, 1], c=labels, cmap=plt.cm.Spectral)
    ax.set_aspect('auto')
    plt.axis('tight')
    plt.title('LLE 2D: {}'.format(metric))
    # ax = plt.subplot(211)
    # ax.scatter(Y[:, 0].real, Y[:, 1].real, c=labels, cmap=plt.cm.Spectral)
    # ax.set_aspect('auto')
    # plt.title('LLE Y 2D: {}'.format(metric))
    plt.show()

def plot2D3D(k = 4, metric = 'minkowski'):
    components = 2
    X_2r, M = main(images, n_neighbors=k, n_components=components, metric=metric)
    components = 3
    X_3r, M = main(images, n_neighbors=k, n_components=components, metric=metric)
    X_2r = X_2r[0]
    X_3r = X_3r[0]

    ax = plt.subplot(212)
    ax.scatter(X_2r[:, 0], X_2r[:, 1], c=labels, cmap=plt.cm.Spectral)
    ax.set_aspect('auto')
    plt.axis('tight')
    plt.title('LLE 2D: {}'.format(metric))

    ax = plt.subplot(211, projection='3d')
    ax.scatter(X_3r[:, 0], X_3r[:, 1], X_3r[:, 2], c=labels, cmap=plt.cm.Spectral)
    ax.set_aspect('auto')
    plt.axis('tight')
    plt.title('LLE 3D: {}'.format(metric))

    plt.show()

def plotDiffDistances(k = 4):
    plot3D(metric='euclidean',k = k)
    plot3D(metric='manhattan', k = k)
    plot3D(metric='minkowski', k = k)

def reconstruct(index = 0, k=4, metric='minkowski'):
    Y, M = main(images, metric=metric)
    Y = Y[0]
    Y_i = Y[index]
    nbrs = NearestNeighbors(n_neighbors=k + 1, n_jobs=1,metric=metric).fit(Y)
    dist,neighbors= nbrs.kneighbors(Y_i, n_neighbors=k+1)
    neighbors = neighbors[0,1:]
    nbrs_emb = Y[neighbors,:]
    W = linalg.lstsq(nbrs_emb.T,Y_i)[0]
    X_nbrs = images[neighbors,:]
    X_r = np.dot(W.T,X_nbrs)
    return X_r

def plot_image(image, index = 0):
    image_reshaped = np.reshape(image,(28,28))
    fig = plt.figure()
    a = fig.add_subplot(1, 2, 1)
    imgplot = plt.imshow(image_reshaped, cmap=plt.cm.Greys)
   # imgplot.set_interpolation('nearest')
    a.set_title('Reconstructed')
    a.xaxis.set_ticks_position('bottom')
    a.yaxis.set_ticks_position('left')
    a = fig.add_subplot(1,2,2)
    imgplot = plt.imshow(np.reshape(images[index,:],(28,28)), cmap=plt.cm.Greys)
    #imgplot.set_interpolation('nearest')
    a.set_title('Orig')
    a.xaxis.set_ticks_position('bottom')
    a.yaxis.set_ticks_position('left')
    plt.show()

def plot_SV(metric='minkowski', k = 4, components = 2 ):
    X_r, M = main(images,n_neighbors=k, n_components=components, metric='euclidean')
    k_skip = 1
    tol = 1E-6
    max_iter = 100
    random_state = check_random_state(None)
    v0 = random_state.uniform(-1, 1, M.shape[0])
    eigen_values, eigen_vectors = eigsh(M, components + k_skip, sigma=0.0,
                                        tol=tol, maxiter=max_iter,
                                        v0=v0)

    eigen_values, eig_vector = linalg.eig(M)
    sorted = np.argsort(eigen_values)
    plt.plot(eigen_values[sorted])
    plt.show()

#plot2D(metric='euclidean', k = 15)

#plot2D3D(metric='euclidean', k = 15)
#plot_SV(metric='euclidean', k = 15, components=7)
#ind = 10
# X_r = reconstruct(index=ind, k = 30, metric = 'euclidean')
#plot_image(X_r, index=ind)

# X_r, M = main(images,n_neighbors=15, n_components=3, metric='euclidean')
# plt.imshow(M)
# plt.show()

plotDiffDistances(k=40)