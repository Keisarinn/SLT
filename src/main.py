from scipy.sparse import eye, csr_matrix
from scipy.linalg import solve, eigh
import struct
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_array
from sklearn.utils.validation import FLOAT_DTYPES
import os

def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])


def load_data(fname_lbl, fname_img):
    print("Loading data...")
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows * cols)
        # img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    return img, lbl

def show_image(image):
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()

def visualize_embedding_2d(Y, labels, name=None):
    import matplotlib.pyplot as plt
    import pandas as pd
    df = pd.DataFrame(dict(x=Y[:,0], y=Y[:,1], label=labels))
    groups = df.groupby('label')

    fig, ax = plt.subplots()
    ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
    for labs, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=1, label=labs)
    ax.legend()
    ax.set(title='2D embedding')
    if(name != None):
        fig.savefig('../results/embedding_' + name + '.png')
        plt.close()

def visualize_embedding_3d(Y, labels, name=None):
    import matplotlib.pyplot as plt
    import pandas as pd
    from mpl_toolkits.mplot3d import Axes3D
    import pylab

    df = pd.DataFrame(dict(x=Y[:,0], y=Y[:,1], label=labels))
    groups = df.groupby('label')

    fig = pylab.figure()
    ax = Axes3D(fig)

    ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
    for labs, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=1, label=labs)
    ax.legend()
    ax.set(title='3D embedding')
    if(name != None):
        plt.savefig('../results/embedding_' + name + '.png')
        plt.close()


def visualize_M(M, dim, name=None):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    sns.heatmap(M[0:dim,0:dim].T)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=10))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(base=10))
    ax.set_title('M matrix')
    if(name != None):
        plt.savefig('../results/M_' + name + '.png')
        plt.close()

def structure_analysis(M, name=None):
    import matplotlib.pyplot as plt
    if( name != None and os.path.exists('../results/singulars_' + name + '.npy')):
        s = np.load('../results/singulars_' + name + '.npy');
    else:
        U, s, V = np.linalg.svd(M);
    plt.plot(s[::-1]);
    plt.ylabel("Singular values")
    plt.title("Singular values of M matrix")
    if(name != None and not os.path.exists('../results/singulars_' + name + '.npy')):
        np.save('../results/singulars_' + name, s)
        plt.savefig('../results/eigens_' + name + '.png')
        plt.close()

def reconstruct(y, Y, X, n_neighbors, metric, reg=1e-3):
    knn = NearestNeighbors(n_neighbors + 1, metric=metric).fit(Y)
    Y = knn._fit_X
    ind = knn.kneighbors(y, return_distance=False)[:, 1:]
    data = barycenter_weights(y, Y[ind], reg=reg)

    return np.dot(data, X[ind,:]).reshape(X.shape[1])



def barycenter_weights(X, Z, reg=1e-3):

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

def barycenter_kneighbors_graph(X, n_neighbors, metric, reg=1e-3):
    print("Fitting weights...")
    knn = NearestNeighbors(n_neighbors + 1, metric=metric).fit(X)
    X = knn._fit_X
    n_samples = X.shape[0]
    ind = knn.kneighbors(X, return_distance=False)[:, 1:]
    data = barycenter_weights(X, X[ind], reg=reg)
    indptr = np.arange(0, n_samples * n_neighbors + 1, n_neighbors)
    return csr_matrix((data.ravel(), ind.ravel(), indptr),
                      shape=(n_samples, n_samples))

def null_space(M, k):
    print("Solving M...")
    k_skip = 1;
    if hasattr(M, 'toarray'):
            M = M.toarray()
    eigen_values, eigen_vectors = eigh( M, eigvals=(k_skip, k + k_skip - 1), overwrite_a=True)
    index = np.argsort(np.abs(eigen_values))
    return eigen_vectors[:, index]

def locally_linear_embedding(X, n_neighbors, dim, metric, name):

    # nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=metric)
    # nbrs.fit(X)

    W = barycenter_kneighbors_graph(X, n_neighbors, metric)
    save_sparse_csr('../results/weights_' + name, W);

    M = (W.T * W - W.T - W).toarray()
    M.flat[::M.shape[0] + 1] += 1  # W = W - I = W - I

    return null_space(M, dim), M

def embede(images, labels, neighbors_n, dims, metrics):
    print("Processing started.")

    #Start processing
    for metric in metrics:
        for k in neighbors_n:
            for dim in dims:
                # Embedding
                name = metric + '_neighs' + str(k) + '_dim' + str(dim);
                embedding, M = locally_linear_embedding(images, k, dim, metric, name)
                np.save('../results/M_' + name, M);
                np.save('../results/embedding_' + name, embedding);

                structure_analysis(M, name);
                if dim == 2:
                    visualize_embedding_2d(embedding, labels, name);
                else:
                    visualize_embedding_3d(embedding, labels, name);
                visualize_M(M, 240, name)


    print("Processing finished.")

def get_line(start, end, n):
    delta = (end - start) / n
    return np.array(list(map( lambda x: np.dot(delta,x) + start, range( 1, n  ) )))

def main():

    #*****************************       Parameters and Data loading      ****************
    # Specify Parameters for embedding, or name for loading and analysis
    neighbors_n = [10];
    dims = [2];
    metrics = ['l1'];
    # name = 'l1_neighs10_dim2';
    n = 3000;

    ## Load data
    images, labels = load_data('../dataset/train-labels-idx1-ubyte', '../dataset/train-images-idx3-ubyte')
    images = images[0:n, :];
    labels = labels[0:n];

    #*****************************       Embedding      ****************
    embede(images, labels, neighbors_n, dims, metrics);


    #*****************************       Analysis      ****************
    #Load the saved result
    # M = np.load('../results/' + str(n) + '/M_' +  name + '.npy');
    # W = load_sparse_csr('../results/' + str(n) + '/weights_' +  name + '.npz');
    # embedding = np.load('../results/' + str(n) + '/embedding_' +  name + '.npy');

    # structure_analysis(M, name);
    # visualize_embedding_2d(embedding, labels);
    # visualize_embedding_3d(embedding, labels);
    # visualize_M(M, 50)

    #*****************************       Reconstruction      ****************
    # ind1 = 19 # Number 9
    # ind2 = 22 # Number 9
    # points = get_line(images[ind1,:], images[ind2,:], 5)
    # show_image(images[ind1,:].reshape(28, 28))
    # show_image(images[ind2,:].reshape(28, 28))
    # for p in points:
    #     show_image(p.reshape(28, 28))
    #
    # points_e = get_line(embedding[ind1,:], embedding[ind2,:], 5)
    # res = reconstruct(embedding[ind1,:], embedding, images, 10, 'l1')
    # show_image(res.reshape(28, 28))
    # res = reconstruct(embedding[ind2,:], embedding, images, 10, 'l1')
    # show_image(res.reshape(28, 28))
    # for p in points_e:
    #     res = reconstruct(p, embedding, images, 10, 'l1')
    #     show_image(res.reshape(28, 28))

main()