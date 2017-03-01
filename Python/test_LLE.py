import LLE_implementation as LLE
import numpy as np

X = np.array([[0,0,0,0],[1,2,3,4],[2,2,3,1],[1,1,1,5]])

x = np.array([3,4])
nj = np.array([1,1])
nk = np.array([1,2])

# print LLE.create_Cjk(x, nj, nk)
#
#
nn = np.array([1,2])
c = LLE.create_C_matrix(0, X, nn, 2)
print c
#
# total = LLE.sum_numerator(c, 1)
# print total
#
# total = LLE.sum_denominator(c)
# print total

# weights = LLE.find_weights(X, n_neighbours=5, norm=1)
# print weights
# print weights.shape
#
# M = LLE.get_M(weights)
# print M
#
# print "eigenvectors"
# eig = LLE.get_eigen(M,2)
# print eig

print "regularizes"
reg = LLE.regularization_matrix(c)
print reg

# y =LLE.fit_LLE(X, 2, 2)
# print y
