# Author: Julien Lamour; lamourj (at) ethz.ch
# Date: March 1, 2017

from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D

N = 5000 # size of the dataset to be used. Max: 60'000
# original dimension D = 28*28 = 784
d = 3 # dimension to map to
k = 10 # number of nearest neighbors to consider

# Load data
mndata = MNIST('./data/')
trainingData = mndata.load_training() # Only using the training dataset since it is already big enough
x = np.array(trainingData[0])[:N]
labels = np.array(trainingData[1])[:N]

# Find neighbours in original space
closestPoints = []
print("Nearest neighbors computation")
for i in range(N):
	dist = np.linalg.norm(x-x[i],axis=1) # distances from Xi to all points
	sortedClosestIndexes = np.argsort(dist)[1:k+1] # returns an array of the indexes of the closest k points to x_i
	closestPoints.append(sortedClosestIndexes)

closestPoints = np.array(closestPoints)

# Solve for reconstruction weights W
W = np.zeros((N,N))

print("Reconstruction weights computation")
for i in range(0,N):
	Z = []
	for neighbourIndex in closestPoints[i]:
		Z.append(np.array(x[neighbourIndex] - x[i]))
	Z = np.array(Z)
	C = np.dot(Z, Z.T)
	w = np.linalg.solve(C,np.ones(k))
	
	w = w / w.sum()

	W[i,closestPoints[i]] = w

# Compute embedding coordinates Y using weights W
M = np.identity(N) - W
M = np.dot(M.T, M)

# M matrix plot building
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_aspect('equal')
fig = plt.imshow(M, interpolation='nearest', cmap=plt.cm.gray)
fig.axes.get_yaxis().set_visible(False)
fig.axes.get_xaxis().set_visible(False)
plt.title('M matrix plot')
plt.colorbar()
plt.show()

# find smallest d+1 eigenvectors of M
print('Eigenvectors computation')
eigenValues,eigenVectors = np.linalg.eig(M)

idx = eigenValues.argsort()
eigenValues = eigenValues[idx]
eigenVectors = eigenVectors[:,idx]
Y = eigenVectors[:,1:d+1]
Y = Y.T

# Singular values plot
print('Singular values computation')
u,singVals,v = np.linalg.svd(M)

fig = plt.figure()
n, bins, patches = plt.hist(singVals, 50, normed=1, facecolor='green', alpha=0.75)
plt.title('M matrix singular values histogram')
plt.show()


# Reduced space plot
if(d==2 or d==3):
	fig = plt.figure()

	if(d==2):
		fig = plt.scatter(Y[0,:], Y[1,:], c=labels)
		fig.axes.get_yaxis().set_visible(False)
		fig.axes.get_xaxis().set_visible(False)
	elif(d==3):
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(Y[0,:], Y[1,:], Y[2,:], c=labels)
	plt.title('Mapping to the lower dimensional space')
	plt.show()