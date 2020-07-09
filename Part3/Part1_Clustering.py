# -*- coding: utf-8 -*-
from matplotlib.pyplot import plot, figure, show, legend, xlabel
from scipy.io import loadmat
from toolbox_02450 import clusterplot
from toolbox_02450 import clusterval
from sklearn import model_selection
from sklearn.cluster import k_means
from updateData_part2 import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
from scipy.linalg import svd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

# Centering and reducing the data
X = sc.fit_transform(X)
y=y.squeeze()

###################### PCA
# PCA by computing SVD of X
U,S,Vh = svd(X,full_matrices=False)
V = Vh.T #transpose the Hermitian of V to obtain the correct V
Z = X @ V #rpoject the centered data onto the principal component space

# Perform hierarchical/agglomerative clustering on data matrix
Method = 'single'
Metric = 'euclidean'

Zlink = linkage(X, method=Method, metric=Metric)

# Compute and display clusters by thresholding the dendrogram
Maxclust = 10
cls = fcluster(Zlink, criterion='maxclust', t=Maxclust)
figure('Clusters from the dendogram with single')
clusterplot(Z, cls.reshape(cls.shape[0],1), y=y)

fig = plt.figure('Clustering in 3D: single')
ax = fig.add_subplot(111, projection='3d')
x = np.array(Z[:,0])
yy = np.array(Z[:,1])
z = np.array(Z[:,2])
ax.scatter(x,yy,z, marker="o", c=cls)
fig.show()
ax.set_xlabel('PC0')
ax.set_ylabel('PC1')
ax.set_zlabel('PC2')

# Display dendrogram
max_display_levels=30
figure('Dendogram', figsize=(10,4))
dendrogram(Zlink, truncate_mode='level', p=max_display_levels)
show()

# Perform hierarchical/agglomerative clustering on data matrix
Method = 'complete'
Metric = 'euclidean'

Zlink = linkage(X, method=Method, metric=Metric)

# Compute and display clusters by thresholding the dendrogram
Maxclust = 10
cls = fcluster(Zlink, criterion='maxclust', t=Maxclust)
figure('Clusters from the dendogram with complete')
clusterplot(Z, cls.reshape(cls.shape[0],1), y=y)

fig = plt.figure('Clustering in 3D: complete')
ax = fig.add_subplot(111, projection='3d')
x = np.array(Z[:,0])
yy = np.array(Z[:,1])
z = np.array(Z[:,2])
ax.scatter(x,yy,z, marker="o", c=cls)
fig.show()
ax.set_xlabel('PC0')
ax.set_ylabel('PC1')
ax.set_zlabel('PC2')

# Display dendrogram
max_display_levels=30
figure('Dendogram', figsize=(10,4))
dendrogram(Zlink, truncate_mode='level', p=max_display_levels)
show()

#### GMM
# Number of clusters
K = 5
cov_type = 'full' # e.g. 'full' or 'diag'

# define the initialization procedure (initial value of means)
initialization_method = 'random'#  'random' or 'kmeans'
# random signifies random initiation, kmeans means we run a K-means and use the
# result as the starting point. K-means might converge faster/better than  
# random, but might also cause the algorithm to be stuck in a poor local minimum 

# type of covariance, you can try out 'diag' as well
reps = 1
# number of fits with different initalizations, best result will be kept
# Fit Gaussian mixture model
gmm = GaussianMixture(n_components=K, covariance_type=cov_type, n_init=reps, 
                      tol=1e-6, reg_covar=1e-6, init_params=initialization_method).fit(X)
cls = gmm.predict(X)    
# extract cluster labels
cds = gmm.means_        
# extract cluster centroids (means of gaussians)
covs = gmm.covariances_
# extract cluster shapes (covariances of gaussians)
if cov_type.lower() == 'diag':
    new_covs = np.zeros([K,M,M])    
    
    count = 0    
    for elem in covs:
        temp_m = np.zeros([M,M])
        new_covs[count] = np.diag(elem)
        count += 1

    covs = new_covs

# Plot results:
#figure(figsize=(14,9))
#clusterplot(X, clusterid=cls, centroids=cds, y=y, covars=covs)
#show()

## In case the number of features != 2, then a subset of features most be plotted instead.
figure('Cluster from the GMM', figsize=(14,9))
idx = [0,1] # feature index, choose two features to use as x and y axis in the plot
clusterplot(Z[:,idx], clusterid=cls, centroids=cds[:,idx], y=y, covars=covs[:,idx,:][:,:,idx])
show()


#plot it in 3D
fig = plt.figure('GMM clustering on data in 3D')
ax = fig.add_subplot(111, projection='3d')
x = np.array(Z[:,0])
yy = np.array(Z[:,1])
z = np.array(Z[:,2])
ax.scatter(x,yy,z, marker="o", c=cls)
fig.show()
ax.set_xlabel('PC0')
ax.set_ylabel('PC1')
ax.set_zlabel('PC2')

# Range of K's to try
KRange = range(1,10)
T = len(KRange)

covar_type = 'full'       # you can try out 'diag' as well
reps = 3                  # number of fits with different initalizations, best result will be kept
init_procedure = 'kmeans' # 'kmeans' or 'random'

# Allocate variables
BIC = np.zeros((T,))
AIC = np.zeros((T,))
CVE = np.zeros((T,))

# K-fold crossvalidation
CV = model_selection.KFold(n_splits=10,shuffle=True)

for t,K in enumerate(KRange):
        print('Fitting model for K={0}'.format(K))

        # Fit Gaussian mixture model
        gmm = GaussianMixture(n_components=K, covariance_type=covar_type, 
                              n_init=reps, init_params=init_procedure,
                              tol=1e-6, reg_covar=1e-6).fit(X)
        
        # Get BIC and AIC
        BIC[t,] = gmm.bic(X)
        AIC[t,] = gmm.aic(X)

        # For each crossvalidation fold
        for train_index, test_index in CV.split(X):

            # extract training and test set for current CV fold
            X_train = X[train_index]
            X_test = X[test_index]

            # Fit Gaussian mixture model to X_train
            gmm = GaussianMixture(n_components=K, covariance_type=covar_type, n_init=reps).fit(X_train)

            # compute negative log likelihood of X_test
            CVE[t] += -gmm.score_samples(X_test).sum()
            
# Plot results

figure('BIC and AIC'); 
plot(KRange, BIC,'-*b')
plot(KRange, AIC,'-xr')
legend(['BIC', 'AIC'])
xlabel('K')
show()

figure('Crossvalidation'); 
plot(KRange, 2*CVE,'-ok')
legend(['Crossvalidation'])
xlabel('K')
show()