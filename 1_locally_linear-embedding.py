from numpy import linalg as LA
import numpy as np
from scipy.spatial.distance import cdist, minkowski
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import LocallyLinearEmbedding as LLE_scikit
import cPickle as pickle


#load images and labels
def load(number, shuffle=True):
    images = pickle.load(open('mnist_images', 'rb'))
    labels = pickle.load(open('mnist_labels','rb'))
    if shuffle==True:
        idx = np.arange(len(images))
        np.random.shuffle(idx)
        images = np.array([images[i] for i in idx[0:number]])
        labels = np.array([labels[i] for i in idx[0:number]])
    return images, labels


#plot images to verify
def plot_images():
    rand = np.random.randint(0,69999)
    plt.imshow(np.reshape(images[rand],(28,28)),cmap='gray')
    test = np.reshape(images[rand],(28,28))
    print labels[rand]
    return test    
    
      
#plot points in 2d    
def plot_points_2d(X,y):
    color = ['#FF0000', '#FF8000', '#80FF00', '#00FFFF', '#7F00FF', '#FF00FF', '#FFFF00', '#404040', '#006666', '#660066']
    for i in range(len(X)):
        plt.scatter(X[i,0],X[i,1], c=color[y[i]])
    plt.show()    
    
#plot points in 3d    
def plot_points_3d(X,y):
    color = ['#FF0000', '#FF8000', '#80FF00', '#00FFFF', '#7F00FF', '#FF00FF', '#FFFF00', '#404040', '#006666', '#660066']
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(X)):
        ax.scatter(X[i,0],X[i,1],X[i,2], c=color[y[i]])
    plt.show()    


#euclidean distance
def dist_euc(a,b):
    return LA.norm(a-b)
    
#minkowski distance
def dist_mink(a,b):
    return minkowski(a,b,3)    

#mahalanobis distance    
def dist_maha(a,b):
    a = np.reshape(a,(28,28))
    b = np.reshape(b,(28,28))
    return cdist(a,b,'mahalanobis') 

#find k nearest neighbours with custom distance measure. returns indexes of k nearest images.
def KNN(X,k,dist):
    knn = np.zeros((len(X),k),dtype=np.uint32)
    for i in range(len(X)):
        d = np.zeros(len(X))
        for j in range(len(X)):
            d[j] = dist(X[i,:],X[j,:])
        
        idx = np.argpartition(d, k+1)[0:k+1]
        idx = np.array([idx[l] for l in range(len(idx)) if idx[l]!=i],dtype=np.uint32)  
        knn[i,:] = idx
        
    return knn
    
    
    
#implementation of LLE (X:data, k:k-nn, d:dim(Y), dist:distance measure for knn)
def LLE(X,k,d,dist):
    knn = KNN(X,k,dist)
    
    #compute W
    W = np.zeros((len(X),len(X)))
    for i in range(len(X)):
        
        C = np.zeros((k,k))
        #compute Ci
        for j in range(k):
            for l in range(k):
                C[j,l] = np.dot(X[i]-X[knn[i,j]], X[i]-X[knn[i,l]])
        #for r in range(len(C)):
            #C[r,r] = C[r,r]*0.001
        #compute wi
        w = LA.lstsq(C,np.ones((len(C),1)))[0]
        #w = LA.solve(C,np.ones((len(C),1)))
        w = w / float(np.sum(w))
        
        #fill w in W at the right places
        for m in range(k):
            W[i,knn[i,m]] = w[m]
        
        
    #compute Y
    Y = np.zeros((len(X),d))
    
    #compute M
    m = np.eye(len(X)) - W
    M = np.dot(np.transpose(m),m)
    
    #perform eig
    eig_vals, eig_vecs = LA.eig(M)
    
    #put lowest eigenvectors into Y
    idx = np.argsort(eig_vals)[1:d+1]
    for s in range(d):
        Y[:,s] = eig_vecs[:,idx[s]]
    
    return Y,M,W 

    

###################### MAIN ##############################
#(code junks that have been used to execute the functions for the different questions)
    
images, labels = load(number=1000)  
    
knn_s = 10
    
########################################## my LLE
Y, M, W = LLE(images,knn_s,3,dist_mink)
#plt.matshow(M,cmap='Greys')
#plt.matshow(M)
#U, s, V = LA.svd(M)
#plt.plot(s[0:10])

plot_points_2d(Y,labels)
plot_points_3d(Y,labels)

########################################## scikit LLE
knn_s = 15
n_components = 2
lle = LLE_scikit(n_neighbors=knn_s, n_components=n_components, neighbors_algorithm='ball_tree', n_jobs=-1)
Y_lle_sk = lle.fit_transform(images)
plot_points_2d(Y_lle_sk,labels)
#plot_points_3d(Y_lle_sk,labels)

################################ linear interpolation in original space
f = np.linspace(0,1,10)
f=f[10]
inter = (1-f)*images[100,:] + f*images[1,:]
plt.imshow(np.reshape(inter,(28,28)),cmap='gray')
