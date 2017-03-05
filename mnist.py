from mnist_load import load_dataset
from scipy import spatial
from scipy.linalg import eigh
from matplotlib import offsetbox
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

X=X_test.reshape([X_test.shape[0],784])
X=X[0:1000,:]
label=y_test[0:1000]
n_neighbor=5

norm=1
d=3

W=np.zeros((X.shape[0],X.shape[0]))
C=np.zeros((X.shape[0],n_neighbor,n_neighbor))
ckdtree = spatial.cKDTree(data=X, leafsize=100)

for i in range(X.shape[0]):
    dist, nbrs_index = ckdtree.query(X[i], k=n_neighbor+1, p=norm) 
    nbrs_index = np.delete(nbrs_index, 0)
    
    nbrs = X[nbrs_index]
    for j in range(n_neighbor):
        for k in range(n_neighbor):
            C[i,j,k] = np.dot(X[i,:]-nbrs[j],X[i,:]-nbrs[k])
            
for i in range(X.shape[0]):
    
    dist, nbrs_index = ckdtree.query(X[i], k=n_neighbor+1, p=norm)
    
    nbrs_index = np.delete(nbrs_index, 0)
    

    invC = np.linalg.inv(C[i])
    
    sum2=0
    for index_l in range(invC.shape[0]):
        for index_k in range(invC.shape[0]):
            sum2 = sum2 + invC[index_l][index_k]
                
    for j in range(n_neighbor):
        index=nbrs_index[j]
        sum1=0
        for index_k in range(invC.shape[0]):
            sum1 = sum1 + invC[j][index_k]
        
        W[i][index] = sum1 / sum2

I_M = np.identity(W.shape[0]) - W
M = np.dot(I_M.transpose(), I_M)
####

#####

eigenvalues, eigenvectors = eigh(M, eigvals_only=False, eigvals=(0,d), turbo=True)
u = np.delete(eigenvectors, (0), axis=1)
y=u

# plot 2d
#if d==2:
#    y_min, y_max = np.min(y, 0), np.max(y, 0)
#    y = (y - y_min) / (y_max - y_min)/1.05
#    plt.figure()
#    ax = plt.subplot(111)
#    for i in range(y.shape[0]):
#        ax.text(y[i, 0], y[i, 1], str(label[i]),
#                 color=plt.cm.Set1(label[i] / 8.),
#                 fontdict={'weight': 'bold', 'size': 9})
#    plt.savefig('test.eps')
#        
#if d==3:
#    y_min, y_max = np.min(y, 0), np.max(y, 0)
#    y = (y - y_min) / (y_max - y_min)
#    plt.figure()
#    ax = plt.subplot(111, projection='3d')
#    for i in range(y.shape[0]):
#        ax.text(y[i, 0], y[i, 1],y[i,2], str(label[i]),
#                 color=plt.cm.Set1(label[i] / 8.),
#                 fontdict={'weight': 'bold', 'size': 9})
#    plt.xticks([]), plt.yticks([])
#    plt.show()

pi=0.8
y_test=pi*y[3]+(1-pi)*y[4]

ckdtreey = spatial.cKDTree(data=y, leafsize=100)

dist, nbrs_index = ckdtreey.query(y_test, k=3, p=norm)
#nbrs_index = np.delete(nbrs_index, 0)
print label[nbrs_index]
Y=y[nbrs_index]
#k=np.linalg.lstsq(Y.transpose(), np.reshape(y[0],(3,1)))[0]
k=np.linalg.solve(Y.transpose(), y_test.transpose())
x_rec=np.dot(X[nbrs_index].transpose(),k)
plt.imshow(np.reshape(x_rec,(28,28)),cmap='Greys')


#pi=0
#y_test=pi*y[0]+(1-pi)*y[1]
#
#ckdtreey = spatial.cKDTree(data=y, leafsize=100)
#
#dist, nbrs_index = ckdtreey.query(y_test, k=n_neighbor, p=norm)
##nbrs_index = np.delete(nbrs_index, 0)
#print label[nbrs_index]
#Y=y[nbrs_index]
#k=np.linalg.lstsq(Y.transpose(), np.reshape(y[0],(3,1)))[0]
##k=np.linalg.solve(Y.transpose(), y_test.transpose())
#x_rec=np.dot(X[nbrs_index].transpose(),k)
#plt.imshow(np.reshape(x_rec,(28,28)),cmap='Greys')
    




