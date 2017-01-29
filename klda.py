import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import make_circles
import scipy.spatial.distance
import scipy.linalg
from sklearn.cross_validation import train_test_split
from sklearn import svm
from scipy import exp
def load_data(filename):
	raw_data=open(filename, 'rb')
	X=np.loadtxt(raw_data)
	print(X.shape)
	return X


def klda(X, y, gamma=10, redu_dim=10):
  #Computation of kernel matrix

  #compute square dist (point to point)
  sq_dist=scipy.spatial.distance.pdist(X,'sqeuclidean')
  #reshape sqare dist
  reshape=scipy.spatial.distance.squareform(sq_dist)
  n=reshape.shape

  X1=X[y==1,:]
  X2=X[y==-1,:]
  n1=X1.shape[0]
  n2=X2.shape[0]
  n=n1+n2
  print "Class 1 samples", n1
  print "Class 2 samples", n2

  #RBF kernel
  K=exp(-reshape*(gamma))

  #Compute M1 and M2
  K1=K[:,y==1]
  K2=K[:,y==-1]


  M1=np.sum(K1,axis=1)/float(n1)
  M2=np.sum(K2,axis=1)/float(n2)


  #Compute M
  M=np.dot((M2-M1),(M2-M1).T)


  #Compute N1 and N2
  I1=np.eye(n1)
  O1=1/float(n1)
  T1=I1-O1
  N1=np.dot(K1,np.dot(T1,K1.T))

  I2=np.eye(n2)
  O2=1/float(n2)
  T2=I2-O2
  N2=np.dot(K2,np.dot(T2,K2.T))
  eps=np.diag(np.repeat(0.0001, n))
  print N1.shape, N2.shape, eps.shape

  N=N1+N2+eps

  Ni=np.linalg.inv(N)

  alpha=np.dot(Ni, (M2-M1))
  M=M2-M1
  alpha2=np.linalg.solve(N, M)
  print "alpha inv", alpha.shape
  print "alpha solv", alpha2.shape
  
  print "shape", K.shape, alpha2.shape
  Z=np.dot(K,alpha2)
  print Z.shape, "result"
  return Z.reshape(Z.shape[0],1)





X=load_data('arcene_train.data')
y=load_data('arcene_train.labels')
X_t=klda(X,y)


X_train, X_test, y_train, y_test = train_test_split(X_t, y, test_size=0.4, random_state=0)

clf = svm.SVC(kernel='rbf', C=1).fit(X_train, y_train)
print "rbfsvm", clf.score(X_test, y_test)

clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
print "linear svm", clf.score(X_test, y_test)

# np.testing.assert_array_almost_equal(X_kpca_inbuilt, X_kpca,decimal=2, err_msg='', verbose=True)

