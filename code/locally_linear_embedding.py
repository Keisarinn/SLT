import numpy as np
import struct
from array import array
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

from scipy.sparse import eye, csr_matrix
import scipy.linalg as linalg
from scipy.linalg import solve
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.arpack import eigsh

from sklearn.manifold import LocallyLinearEmbedding

from skimage.measure import compare_ssim as ssim
from scipy.ndimage.filters import gaussian_filter

#Set font size of plots
import matplotlib 
matplotlib.rcParams.update({'font.size': 22})


"""
Constants
"""

neighbor_method = 'mse' #'mse' or 'ssim'

data_folder = "../data/"
train_img_path = data_folder+"train-images.idx3-ubyte"
train_lbl_path = data_folder+"train-labels.idx1-ubyte"

plot_folder = "../plots/"
save_figures = True #Automatically save figures


"""
Load function
Cherry picked from 
https://github.com/sorki/python-mnist/blob/master/mnist/loader.py
"""
def load(path_img, path_lbl):
    with open(path_lbl, 'rb') as file:
        magic, size = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError('Magic number mismatch, expected 2049,'
                             'got {}'.format(magic))

        labels = array("B", file.read())

    with open(path_img, 'rb') as file:
        magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError('Magic number mismatch, expected 2051,'
                             'got {}'.format(magic))

        image_data = array("B", file.read())

    images = []
    for i in range(size):
        images.append([0] * rows * cols)

    for i in range(size):
        images[i][:] = image_data[i * rows * cols:(i + 1) * rows * cols]

    return images, labels
    

#Load data
train_data, train_labels = load(train_img_path, train_lbl_path)


"""
LLE method
"""

def locally_linear_embedding(X, n_neighbors, out_dim, tol=1e-3, max_iter=200):
    
    n = len(X)
    
    """
    Compute Neighbors
    """
    N = np.zeros((n,n_neighbors),dtype=np.int)
    if neighbor_method == 'ssim':
        #A more complex neighbor computation based on structural similarity
        #The images are smoothed with a Gaussian filter before comparison
        for i in range(n):
            imgA = gaussian_filter(np.reshape(X[i],(28,28)),1.0)/255.0
            similarities = np.zeros(n)
            for j in range(n):
                imgB = gaussian_filter(np.reshape(X[j],(28,28)),1.0)/255.0
                similarities[j] = ssim(imgA,imgB,win_size=5)
            N[i,:] = np.argsort(similarities)[-n_neighbors-1:-1]
            if i % 50==0:
                print("Progress: "+str(i)+"/"+str(n))
    else:
        knn = NearestNeighbors(n_neighbors + 1).fit(X) 
        N = knn.kneighbors(X, return_distance=False)[:, 1:]
    
    """
    Compute Weights
    """
    Wdense = np.zeros((n,n_neighbors))
    for i in range(n):
        z = X[i] - X[N[i,:]]
        C = z.dot(z.T)
        C = C + eye(n_neighbors)*tol*np.trace(C)
        w = solve(C,np.ones(n_neighbors), sym_pos=True)
        Wdense[i,:] = w/np.sum(w)
        
    indptr = np.arange(0, n * n_neighbors + 1, n_neighbors)
    W = csr_matrix((Wdense.ravel(), N.ravel(),indptr), shape=(n, n))

    """
    Compute M = (I-W)' (I-W)
    """
    M = eye(*W.shape, format=W.format) - W
    M = (M.T).dot(M).tocsr()


    """
    Plot Matrix M
    """
    Mvisu = M.todense()
    Mvisu[Mvisu==0] = np.nan
    fig, ax = plt.subplots(figsize=(12,12))
    ax.matshow(Mvisu,cmap=plt.cm.gray)
    ax.set_title("Matrix M")
    plt.draw()


    """
    Plot singular values of M
    """    
    u,s,v = np.linalg.svd(M.todense())
    s = list(reversed(s))
    fig, ax = plt.subplots(figsize=(12,8))
    plt.title("Sorted singular values of M")
    plt.ylabel("Value")
    ax.plot(list(range(0,len(s))),s)
    axins = inset_axes(ax,5,3,loc=2)
    axins.plot(list(range(0,len(s))),s)
    x1,x2,y1,y2 = 0, 20, 0, 0.1
    axins.set_xlim(x1,x2)
    axins.set_ylim(y1,y2)
    axins.yaxis.tick_right()
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    
    plt.draw()


    """
    Compute eigenvalues and eigenvectors on sparse matrix
    """
    eigen_values, eigen_vectors = eigsh(M, out_dim + 1, sigma=0.0,
                                                tol=tol, maxiter=max_iter)
    index = np.argsort(eigen_values) #In case the values are not sorted
    index = index[1:] # Drop lowest eigenvalue
    return eigen_vectors[:, index]


"""
Reconstruction
"""


def simple_reconstruct(Y,X,p,n_neighbors,pic):
    #Y embedded coordinates
    #X original vectors
    #p new point
    
    knn = NearestNeighbors(n_neighbors).fit(Y) 
    d, N = knn.kneighbors(p.reshape(1,-1), return_distance=True)

    w = linalg.solve(Y[N,:][0].T, p.reshape(-1,1)).T
    w = w/np.sum(w)
#    print(d, w)
 
    pX = w.dot(X[N,:])

    img = np.reshape(pX,(28,28))
    fig = plt.figure()
    plt.imshow(img,cmap='gray')
    if save_figures:
        fig.savefig(plot_folder+"series/"+str(pic)+".png")
    plt.draw()


def learn_mapping(X,Y,ps):
    #Learns reverse mapping from Y to X
    
    X = X/255.0 #Normalize X by maximum value
    
    import tensorflow as tf
    num_epochs = 50
    n = np.shape(Y)[0]
    din = np.shape(Y)[1]    
    dout = np.shape(X)[1]
    
    neurons_h1 = 256
    neurons_h2 = 512
    
    source = tf.placeholder(tf.float32, shape=[1,din])
    target = tf.placeholder(tf.float32, shape=[1,dout])
    keep_prob = tf.placeholder(tf.float32)
    
    W_h1 = tf.Variable(tf.truncated_normal(shape=[din,neurons_h1]))
    b_h1 = tf.Variable(tf.constant(shape=[neurons_h1],value=0.1))    
    o_h1 = tf.nn.relu(tf.matmul(source,W_h1) + b_h1)
    o_h1_drop = tf.nn.dropout(o_h1, keep_prob)
    
    
    
    W_h2 = tf.Variable(tf.truncated_normal(shape=[neurons_h1,neurons_h2]))
    b_h2 = tf.Variable(tf.constant(shape=[neurons_h2],value=0.1))    
    o_h2 = tf.nn.relu(tf.matmul(o_h1_drop,W_h2) + b_h2)
    o_h2_drop = tf.nn.dropout(o_h2, keep_prob)
    
    W_out = tf.Variable(tf.truncated_normal(shape=[neurons_h2,dout]))
    b_out = tf.Variable(tf.constant(shape=[dout],value=0.1))
    
    output = tf.matmul(o_h2_drop, W_out) + b_out
    
    cost = tf.losses.mean_squared_error(target,output)
    train_step = tf.train.AdamOptimizer(0.001).minimize(cost)
    
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("Start training model for reverse mapping")
        for e in range(num_epochs):
            for i in np.random.permutation(n):
                new_source = np.reshape(Y[i,:], [1,-1])
                new_target = np.reshape(X[i,:], [1,-1])
                train_step.run(session=sess, feed_dict={source: new_source, target: new_target, keep_prob: 0.5})
                if i % 500 == 0:
                    print("Cost for sample "+str(i)+":",sess.run([cost],feed_dict={source: new_source, target: new_target, keep_prob: 1.0}))
        print("Training done")
        
        
        #Some samples to compare
        for i in [0,100,356,432,768,862]:
            new_source = np.reshape(Y[i,:], [1,-1])
            generated_image = sess.run([output], feed_dict={source:new_source, keep_prob: 1.0})
            plt.figure()
            plt.imshow(np.reshape(generated_image,(28,28)),cmap='gray')
            plt.show()
            plt.figure()
            plt.imshow(np.reshape(X[i,:],(28,28)),cmap='gray')
            plt.show()
        
        for p in ps:
            new_source = p.reshape(1,-1)
            generated_image = sess.run([output], feed_dict={source:new_source, keep_prob: 1.0})
            print(np.reshape(generated_image,(-1))[100:110])
            plt.figure()
            plt.imshow(np.reshape(generated_image,(28,28)),cmap='gray')
            plt.show()



"""
Sort samples by labels in order to make block structure in matrix visible
"""

samples = 2000
neighbors = 10

#Sort train data for better visuals on block structures
train_data = train_data[0:samples]
train_labels = train_labels[0:samples]
train_data = np.asarray([x for (y,x) in sorted(zip(train_labels,train_data))])
train_labels = sorted(train_labels)

"""
Compute LLE
We can compute the LLE once for 3 dimensions. 
The lower 2 dimensions are not influenced by the fact that we already computed the 3rd one.
I.e. the matrix M is the same independent of how many dimensions the embedded space has.
"""
lle = locally_linear_embedding(train_data, neighbors, 100)

"""
Plot 2D
"""
fig = plt.figure(figsize=(12,8))
plt.title("2D Embedding")
plt.scatter(lle[:,0],lle[:,1],c=train_labels)
if save_figures:
    fig.savefig(plot_folder+'2dEmbeddingNew.png')
plt.draw()


reconstruction_method = 'learn'#'interpolation' or 'learn'

dim = 100 #Dimensions used for reconstruction
a, b = lle[0,:dim], lle[700,:dim]
a, b = np.asarray(a), np.asarray(b)
plt.plot([a[0],b[0]],[a[1],b[1]])
plt.draw()

ps = [] #All points along the path
for t in np.arange(0,1.01,0.1): 
    ps.append((1-t)*a+t*b)

if reconstruction_method == 'interpolation':
    pic = 0 #Used for saving different pictures -> gif
    for p in ps:
        simple_reconstruct(lle[:,0:dim],train_data,p,dim,pic)
        pic = pic+1
elif reconstruction_method == 'learn':
    learn_mapping(train_data,lle[:,:dim],ps)


"""
Plot 3D
"""
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("3D Embedding")
ax.scatter(lle[:,0],lle[:,1],lle[:,2],c=train_labels)
if save_figures:
    fig.savefig(plot_folder+'3dEmbeddingNew.png')
plt.show()






#Sci-kit learn verification to double check custom implementation
show_sklearn_results = False

if show_sklearn_results:
    #2D
    lle2dsk = LocallyLinearEmbedding(n_components=2, n_neighbors=neighbors)
    
    print("Fitting sklearn LLE 2D")
    lle2dsk.fit(train_data[0:samples])
    
    
    train_2d = lle2dsk.transform(train_data)
    
    print(np.shape(train_2d))
    plt.scatter(train_2d[:,0],train_2d[:,1],c=train_labels)
    plt.draw()
    
    # 3D
    lle3dsk = LocallyLinearEmbedding(n_components=3, n_neighbors=neighbors)
    
    print("Fitting sklearn LLE 3D")
    lle3dsk.fit(train_data[0:samples])
    
    
    train_3d = lle3dsk.transform(train_data)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(train_3d[:,0],train_3d[:,1],train_3d[:,2],c=train_labels)
    plt.show()