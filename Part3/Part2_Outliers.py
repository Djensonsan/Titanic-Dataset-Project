import numpy as np
from matplotlib.pyplot import figure, subplot, hist, bar, title, plot, show
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
from sklearn.neighbors import NearestNeighbors
from toolbox_02450 import gausKernelDensity
from updateData import *

#Normalizing the data
X = sc.fit_transform(X)

Nout=10 #set the number of possible outliers to display

########## Gaussian Kernel density by leave-one-out cross-validation
widths = 2.0**np.arange(-5,2)
logP = np.zeros(np.size(widths))
for i,w in enumerate(widths):
    f, log_f = gausKernelDensity(X, w)
    logP[i] = log_f.sum()
val = logP.max()
ind = logP.argmax()

width=widths[ind]
print('Optimal estimated width is: {0}'.format(width))

# Estimate density for each observation not including the observation
# itself in the density estimate
density, log_density = gausKernelDensity(X, width)
help

# Sort the densities
index_order1 = (density.argsort(axis=0)).ravel()
density = density[index_order1]

# Display the index of the lowest densities data objects
print('{0} lowest densities according to Gaussian Kernel density'.format(Nout))
for j in range(Nout): 
    print('{0} lowest density: {1} for data object: {2}'.format(j+1,density[j],index_order1[j]))

# Plot density estimate of outlier score
figure(1)
bar(range(Nout),density[:(Nout)].reshape(-1,))
title('Density estimate')
figure(2)
plot(logP)
title('Optimal width')
show()



######## Nearest neighbors
x = np.linspace(-3, 6, 10)
xe = np.linspace(-10, 10, 712)
K = 20 # Number of neighbors

# Find the k nearest neighbors
knn = NearestNeighbors(n_neighbors=K).fit(X)
D, iN = knn.kneighbors()

# Compute the KNN density
knn_density = 1./(D.sum(axis=1)/K)

# Sort the densities
knn_density_sorted=knn_density
knn_density_sorted = knn_density_sorted[:,np.newaxis]
index_order2 = (knn_density_sorted.argsort(axis=0)).ravel()
knn_density_sorted = knn_density_sorted[index_order2]

# Display the index of the lowest densities data objects
print('\n{0} lowest densities according to Nearest neighbors'.format(Nout))
for j in range(Nout): 
    print('{0} lowest density: {1} for data object: {2}'.format(j+1,knn_density_sorted[j],index_order2[j]))
    
# Plot KNN density
figure(figsize=(6,7))
subplot(2,1,1)
hist(X,x)
title('Data histogram')
subplot(2,1,2)
plot(xe, knn_density)
title('KNN density')
show()


#KNN average relative density (ARD)
# Compute the average relative density
DX, iX = knn.kneighbors(X)
knn_densityX = 1./(DX[:,1:].sum(axis=1)/K)
knn_avg_rel_density = knn_density/(knn_densityX[iX[:,1:]].sum(axis=1)/K)

# Sort the densities
ard_sorted=knn_avg_rel_density
ard_sorted = ard_sorted[:,np.newaxis]
index_order3 = (ard_sorted.argsort(axis=0)).ravel()
ard_sorted = ard_sorted[index_order3]

# Display the index of the lowest densities data objects
print('\n{0} lowest densities according to ARD'.format(Nout))
for j in range(Nout): 
    print('{0} lowest density: {1} for data object: {2}'.format(j+1, ard_sorted[j],index_order3[j]))

# Plot KNN average relative density
figure(figsize=(6,7))
subplot(2,1,1)
hist(X,x)
title('Data histogram')
subplot(2,1,2)
plot(xe, knn_avg_rel_density)
title('KNN average relative density')
show()

#display all the data side to side
print('{0} lowest densities according to Gaussian Kernel density'.format(Nout))
for j in range(Nout): 
    print('{0} & {1} & {2} & {3} & {4} & {5} & {6}'.format(j+1, density[j] ,index_order1[j], np.round(knn_density_sorted[j], 3),index_order2[j], np.round(ard_sorted[j],3),index_order3[j]))

####### Compare found outliers
Nout=20
index_order1 = index_order1[:Nout]
index_order2 = index_order2[:Nout]
index_order3 = index_order3[:Nout]
similar_indexes=[]
print('\nIndex of observations that are found to be outliers by 2 or more methods')
for item in range(Nout):
    if (np.isin(index_order1[item],index_order2)) :
        similar_indexes.append(index_order1[item])
    if (np.isin(index_order1[item],index_order2)) and (index_order1[item] not in similar_indexes):
        similar_indexes.append(index_order1[item])
    if (np.isin(index_order2[item],index_order3)) and (index_order2[item] not in similar_indexes):
        similar_indexes.append(index_order2[item])
print(similar_indexes)


