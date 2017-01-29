import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import make_circles
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.cross_validation import cross_val_score


import scipy.spatial.distance
import scipy.linalg
import sklearn.metrics.pairwise
from scipy import exp
import sklearn.preprocessing
def load_data(filename):
	raw_data=open(filename, 'rb')
	X=np.loadtxt(raw_data)
	print(X.shape)
	return X


def kpca(X, gamma=10, redu_dim=10):
  #Computation of kernel matrix

  #compute square dist (point to point)
  sq_dist=scipy.spatial.distance.pdist(X,'sqeuclidean')
  #reshape sqare dist
  reshape=scipy.spatial.distance.squareform(sq_dist)
  n=reshape.shape
  n=n[0]
  #RBF kernel
  K=exp(-reshape*(gamma))
  #Center Kernel
  oneN=np.ones((n,n))/n
  #K=sklearn.metrics.pairwise.rbf_kernel(X, gamma=10)
  kern_cent=sklearn.preprocessing.KernelCenterer()
  K=kern_cent.fit_transform(K)
  
  #K_cent=K-np.dot(oneN,K)-np.dot(K,oneN)+np.dot(np.dot(oneN,K),oneN)
  #K=K_cent
  
  #Eigen vecs and vals of K
  #eigenval, eigenvec = scipy.linalg.eigh(K)
  eigenval, eigenvec = scipy.linalg.eigh(K,eigvals=(K.shape[0]-redu_dim, K.shape[0]-1))
  idx = np.argsort((eigenval))
  print K.shape
  eigenvec=eigenvec[:,idx]
  eigenval=eigenval[idx]
  # Get highest eigen values and their corresponding highest eigenvalues.
  W=[]
  L=[]
  for i in xrange(1,redu_dim+1):
    W.append(eigenvec[:,-i])
    L.append(eigenval[-i])

  alphas=np.column_stack(W)
  lambdas=np.array(L)
  X_transformed=alphas*np.sqrt(lambdas)
  return (alphas,X_transformed)

X=load_data('arcene_train.data')
y=load_data('arcene_train.labels')


print "==============================For 10 components================================="
#inbuilt kpca
pcamod=KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10, n_components=10)
ib_pca=pcamod.fit_transform(X)

#our kpca
o_pca, o_x =kpca(X,10,10)


#To show our implementation is same as inbuilt
np.testing.assert_array_almost_equal(ib_pca, o_pca,decimal=2, err_msg='', verbose=True)


X_train, X_test, y_train, y_test = train_test_split(o_pca, y, test_size=0.4, random_state=0)

clf = svm.SVC(kernel='rbf', C=1).fit(X_train, y_train)
print "RBF svm,  RBF pca score:", clf.score(X_test, y_test) 

#clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
#print "Linear svm, RBF pca score :", clf.score(X_test, y_test) 


#o_pca, o_x =kpca(X,1,10)
pcamod=KernelPCA(kernel="linear", fit_inverse_transform=True, n_components=10)
ib_pca=pcamod.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(ib_pca, y, test_size=0.4, random_state=0)

clf = svm.SVC(kernel='rbf', C=1).fit(X_train, y_train)
print "RBF svm,  linear pca score:", clf.score(X_test, y_test) 

clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
print "Linear svm, linear pca score :", clf.score(X_test, y_test) 




print "==============================For 99 components================================="
#inbuilt kpca
pcamod=KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10, n_components=99)
ib_pca=pcamod.fit_transform(X)

#our kpca
o_pca, o_x =kpca(X,10,99)

#To show our implementation is same as inbuilt
np.testing.assert_array_almost_equal(ib_pca, o_pca,decimal=2, err_msg='', verbose=True)



X_train, X_test, y_train, y_test = train_test_split(o_pca, y, test_size=0.4, random_state=0)

clf = svm.SVC(kernel='rbf', C=1).fit(X_train, y_train)
print "RBF svm,  RBF pca score:", clf.score(X_test, y_test) 

clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
print "Linear svm, RBF pca score :", clf.score(X_test, y_test) 


#o_pca, o_x =kpca(X,1,99)
pcamod=KernelPCA(kernel="linear", fit_inverse_transform=True, n_components=99)
ib_pca=pcamod.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(ib_pca, y, test_size=0.4, random_state=0)

clf = svm.SVC(kernel='rbf', C=1).fit(X_train, y_train)
print "RBF svm,  linear pca score:", clf.score(X_test, y_test) 

clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
print  cross_val_score(clf, ib_pca, y, cv=5)
print "Linear svm, linear pca score :", clf.score(X_test, y_test) 


