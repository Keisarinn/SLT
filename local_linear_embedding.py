
# coding: utf-8

# ## (a) Get the data

# In[14]:

from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt
import numpy as np
import numpy.random
mnist = fetch_mldata('MNIST original', data_home='~/Documents/ETHZ/2017_Spring/Statistical Learning Theory/slt-coding-exercises/')
choice =  np.random.choice(mnist.data.shape[0], 2000, replace=False)
X = mnist.data[choice].astype(int)
y = mnist.target[choice].astype(int)

# ## (b) LLE implementation

# In[2]:

from numpy.linalg import inv
import sklearn.metrics.pairwise
import heapq
N = X.shape[0]
K = 5
neighbors = np.zeros((N, K), dtype=int)
#D = sklearn.metrics.pairwise.pairwise_distances(X)
#print D

D = np.zeros((N, N), dtype=int)
for i in range(N):
    for j in range(i + 1, N):
        d = (X[i] - X[j])
        D[i][j] = d.dot(d)
        D[j][i] = D[i][j]
print D

# In[3]:

W = np.zeros((N, N))
for i in range(N):
    C = np.zeros((K, K))
    neighbors[i,:] = np.array(sorted(range(N), key=lambda p: D[i][p])[1:K+1])
    #neighbors[i] = list(heapq.nlargest(K, range(N), key=lambda p: D[i][p]))
    for _j, j in enumerate(neighbors[i]):
        for _k, k in enumerate(neighbors[i]):
            d_ij = (X[i] - X[j])
            d_ik = (X[i] - X[k])
            C[_j][_k] = d_ij.dot(d_ik)
    C_I = inv(C)
    for _j, j in enumerate(neighbors[i]):
        W[i][j] = float(np.sum(C_I[_j])) / np.sum(C_I)
print np.sum(W,axis=1)

# In[29]:

from numpy import linalg as LA
d = 2
M = (np.identity(N)-W).T.dot(np.identity(N)-W)
%pylab inline
pylab.rcParams['figure.figsize'] = (10, 10)
plt.matshow(M, cmap=plt.cm.gray)
plt.show()
print M
w, v = LA.eig(M)
# M is positive definite, so the smallest eigenvalue is the one that's supposed to be zero
Y = v[sorted(range(N), key=lambda i: w[i])[1 : d + 1]].real.T

# ## MISC.

# In[36]:

pylab.rcParams['figure.figsize'] = (10, 3)
plt.plot(w, linewidth=2)
plt.show()
plt.plot(list(reversed(sorted(w))), linewidth=2)
plt.show()

# In[36]:

def plot_embedding(X, y, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    if title is not None:
        plt.title(title)
        
from mpl_toolkits.mplot3d import Axes3D
def plot_embedding_3d(X, y, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111, projection='3d')
    for i in range(X.shape[0]):
        #plt.text(X[i, 0], X[i, 1], X[i, 2], str(y[i]), color=plt.cm.Set1(y[i] / 10.), fontdict={'weight': 'bold', 'size': 9})
        ax.scatter(X[i, 0], X[i, 1], X[i, 2], c=plt.cm.Set1(y[i] / 10.), marker='o', alpha=0.7, linewidths=0)
    if title is not None:
        plt.title(title)

# In[6]:

plot_embedding(Y, y, "Locally Linear Embedding of the digits")
plt.show()

# In[37]:

from sklearn import manifold
pylab.rcParams['figure.figsize'] = (6, 4)
clf = manifold.LocallyLinearEmbedding(n_neighbors=10, n_components=2,
                                      method='standard')
X_lle = clf.fit_transform(X)
plot_embedding(X_lle, y, "with 10 nearest neighbors")
plt.show()

# In[17]:


clf = manifold.LocallyLinearEmbedding(n_neighbors=30, n_components=3,
                                      method='standard')
X_lle = clf.fit_transform(X)
plot_embedding_3d(X_lle, y, "with 30 nearest neighbors")
plt.show()

# In[23]:

from skimage.util import view_as_windows
def get_windowed_histogram(img_2d, WINDOW_SIZE = 4):
    STEP_SIZE = WINDOW_SIZE / 2
    NUM_BINS = 8
    GLOBAL_MIN = 0
    GLOBAL_MAX = 255
    all_windows = view_as_windows(img_2d, (WINDOW_SIZE, WINDOW_SIZE), STEP_SIZE)
    flattened_windows = np.reshape(all_windows, (-1, WINDOW_SIZE, WINDOW_SIZE))
    histogram_feature_list = []
    i = 1
    for window in flattened_windows:
        single_hist = np.histogram(window, bins=NUM_BINS, range=(GLOBAL_MIN, GLOBAL_MAX))[0]
        histogram_feature_list.append(single_hist)
    return np.hstack(histogram_feature_list)

# In[30]:

from sklearn import manifold
clf = manifold.LocallyLinearEmbedding(n_neighbors=10, n_components=2,
                                      method='standard')
X_lle = clf.fit_transform(X_hist)
plot_embedding(X_lle, y, "histogram with 10 nearest neighbors")
plt.show()

# In[32]:

plt.imshow(X[0].reshape((28,28)), cmap='gray_r')
plt.show()

# In[38]:

from sklearn import manifold
pylab.rcParams['figure.figsize'] = (6, 4)
clf = manifold.LocallyLinearEmbedding(n_neighbors=10, n_components=2,
                                      method='standard')
X_lle = clf.fit_transform(X)

# In[55]:

from numpy.linalg import inv
Y_new = np.array([[1,0],[0,1],[0,0],[1,1],[0.5,0.5]])
N = len(X_lle)
K = 10
for y_i in Y_new:
    print y_i
    distance = np.zeros(N)
    C = zeros((K,K))
    for i in range(N):
        distance[i] = (y_i - X_lle[i]).dot(y_i - X_lle[i])
    neighbors = sorted(range(N), key=lambda p: distance[p])[:K]
    print y[neighbors]
    print neighbors
    for _j, j in enumerate(neighbors):
        for _k, k in enumerate(neighbors):
            d_ij = (y_i - X_lle[j])
            d_ik = (y_i - X_lle[k])
            C[_j][_k] = d_ij.dot(d_ik)
    C /= np.sum(C)
    C_I = inv(C)
    X_i = np.zeros(len(X[0]))
    for _j, j in enumerate(neighbors):
        w_j = float(np.sum(C_I[_j])) / np.sum(C_I)
        print w_j
        X_i +=  w_j * X[j]
    plt.imshow(X_i.reshape((28,28)), cmap='gray_r')
    plt.show()

# In[45]:

np.array([1,2,3]) + 3*np.array([1,2,3])

# In[ ]:


