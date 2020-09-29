import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat
def pca(X):
    # normalize the features
    X = (X - X.mean()) / X.std()
    
    # compute the covariance matrix
    X = np.matrix(X)
    cov = (X.T * X) / X.shape[0]
    # print(cov)
#    print('cov \n', cov)
#    print()
    # perform SVD
    U, S, V = np.linalg.svd(cov) # singular value decomposition
    # print(U)
    return U, S, V

def project_data(X, U, k):
    U_reduced = U[:,:k]
    return np.dot(X, U_reduced)



def recover_data(Z, U, k):
    U_reduced = U[:,:k]
    return np.dot(Z, U_reduced.T)


faces = loadmat('D:\\ex7faces.mat')
X = faces['X']
print(X.shape)
# plt.imshow(X)


# show one face
face = np.reshape(X[401,:], (32, 32))
# print(face.shape)
# plt.imshow(face)


U, S, V = pca(X)
Z = project_data(X, U, 500)
print(Z.shape)

X_recovered = recover_data(Z, U, 500)
print(X_recovered.shape)
face = np.reshape(X_recovered[401,:], (32, 32))
plt.imshow(face)
