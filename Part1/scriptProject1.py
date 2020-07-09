#This script uses Pclass, Sex, Age, SibSp, Parch, Fare and Embarked to explore the data and compute a PCA.
#Basically, the other columns are deleted, and then all the lines containing NaN values are deleted as well.
#Pclass, Sex and Embarked are coded using a one-out-of-K coding.

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
from scipy.linalg import svd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from matplotlib.pyplot import boxplot, xticks, ylabel, title, show
import pylab 
import scipy.stats as stats


# Load the data
filename = './train.csv'
df = pd.read_csv(filename) #panda returns a dataframe

# Deleting the columns and lines that are not interesting
df = df.drop("PassengerId", axis=1)
df = df.drop("Name", axis=1)
df = df.drop("Ticket", axis=1)
df = df.drop("Cabin", axis=1)
df = df.dropna() #gets rid of the rows containing nan values

# Store the "survived" status in an array apart from the rest of the data
survived = np.asarray(df); #saves all the dataframe status in an numpyarray
survived = survived[:, 0] #keeps only the "survived" column
df = df.drop("Survived", axis=1)

raw_data = np.asarray(df); #converting the dataframe into a numpyarray

# Creating the data matrix X
cols= range(0,7) #select the indexes of the columns to be used
X = raw_data[:,cols] #making the data matrix X
rowSize, colSize = X.shape


# Change "male" and "female" to 1 and 0. for reference : X[row][col]
for i in range(rowSize): 
    if X[i,1]=="male":
        X[i,1]=1;
    elif X[i,1]=="female":
        X[i,1]=0;
        
# Change the Embarked status
for i in range(rowSize): 
    if X[i,-1]=="S":
        X[i,-1]=0;
    elif X[i,-1]=="Q":
        X[i,-1]=1;
    elif X[i,-1]=="C":
        X[i,-1]=2;


X = np.asarray(X.astype(float))


# Calculate and display correlation matrix
corr = np.corrcoef(X, rowvar=False)
corr = corr.round(2)
labels = ["Pclass", "sex", "age", "Sibsp", "Parch", "Fare", "Embarked"]

fig, ax = plt.subplots()
im = ax.imshow(corr, aspect= 'equal', cmap='Oranges', origin='upper', extent=(-0.5, 6.5, -0.5, 6.5))
ax.set_xticks(np.arange(colSize))
ax.set_yticks(np.arange(colSize))
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)

plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(colSize):
    for j in range(colSize):
        text = ax.text(j, i, corr[i, j], ha="center", va="center", color="k", size="9")

ax.set_title("Correlation of the different variables")
fig.tight_layout()
fig.canvas.set_window_title('Correlation')
plt.show()

# Calculate and display covariance matrix
cov = np.cov(X, rowvar=False)
cov = cov.round(2)

fig, ax = plt.subplots()
im = ax.imshow(cov, aspect='equal', cmap='RdYlBu_r', vmin=-25, vmax=25, origin='upper', extent=(-0.5, 6.5, -0.5, 6.5))
ax.set_xticks(np.arange(colSize))
ax.set_yticks(np.arange(colSize))
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)

plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(colSize):
    for j in range(colSize):
        text = ax.text(j, i, cov[i, j], ha="center", va="center", color="k", size="9")

ax.set_title("Covariance of the different variables")
fig.tight_layout()
fig.canvas.set_window_title('Covariance')
plt.show()


#Encode the one-out-of-K values
label_encoder = LabelEncoder()
onehot_encoder = OneHotEncoder(categories = 'auto',sparse=False)

# Code Pclass values
pclass = X[:,0];
onehot_pclass_encoded = onehot_encoder.fit_transform(pclass.reshape(len(pclass),1))


# Code sex values : X[row][col]
sex = X[:,1];
integer_sex_encoded = label_encoder.fit_transform(sex)
integer_sex_encoded = integer_sex_encoded.reshape(len(integer_sex_encoded),1)
onehot_sex_encoded = onehot_encoder.fit_transform(integer_sex_encoded)


# Code embarked status
embarked = X[:,-1];
integer_embarked_encoded = label_encoder.fit_transform(embarked)
integer_embarked_encoded = integer_embarked_encoded.reshape(len(integer_embarked_encoded),1)
onehot_embarked_encoded = onehot_encoder.fit_transform(integer_embarked_encoded)


#replacing the columns with the encoded values

A = onehot_pclass_encoded
A = np.hstack((A, onehot_sex_encoded))
r = np.reshape(X[:,2], (712, 1))
A = np.hstack((A, r))
r = np.reshape(X[:,3], (712, 1))
A = np.hstack((A, r))
r = np.reshape(X[:,4], (712, 1))
A = np.hstack((A, r))
r = np.reshape(X[:,5], (712, 1))
A = np.hstack((A, r))
A = np.hstack((A, onehot_embarked_encoded))
X=np.asarray(A);


# Extracting the names of the attributes
attributeNames = np.asarray(df.columns[cols])

# Turn the data matrix in an array of floats, now that all the data has been changed to floats
X = np.asarray(X.astype(float))

#centering and reducing the data
Y = sc.fit_transform(X)

#or just centering -> doesn't work well
#Y = X - np.ones((rowSize,1))*X.mean(axis=0)

#print(np.mean(Y,axis=0)) #check that all the means are 0
#print(np.std(Y,axis=0,ddof=0)) #check that all the standard deviations are 0

# PCA by computing SVD of Y
U,S,Vh = svd(Y,full_matrices=False)
V = Vh.T #transpose the Hermitian of V to obtain the correct V
Z = Y @ V #rpoject the centered data onto the principal component space

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 
threshold = 0.9

# Plot variance explained
plt.figure(num='Variance explained')
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()

# Plot PCA of the data
f = plt.figure(num='Projection on the first 2 PC')
plt.title('Titanic data: PCA')
# Indices of the principal components to be plotted
i = 0
j = 1
for c in range(2): #2 because it is the number of different values of "survived"
    class_mask = survived==c
    plt.plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)
plt.legend(['Died', 'Survived'])
plt.xlabel('PC{0}'.format(i+1))
plt.ylabel('PC{0}'.format(j+1))
plt.show()

# Printing the PC values
print('PC1:')
print(V[:,0].round(2).T)

print('PC2:')
print(V[:,1].round(2).T)

print('PC3:')
print(V[:,2].round(2).T)

# Plotting the data with the 3 first principal components
fig = plt.figure(num='Projection on the first 3 PC')
ax = fig.add_subplot(111, projection='3d')
i = 0
j = 1
k = 2
for c in range(2): #2 because it is the number of different values of "survived"
    class_mask = survived==c
    ax.scatter(Z[class_mask,i], Z[class_mask,j],Z[class_mask,k], 'o', alpha=.5)
plt.legend(['Died', 'Survived'])
ax.set_xlabel('PC{0}'.format(i+1))
ax.set_ylabel('PC{0}'.format(j+1))
ax.set_zlabel('PC{0}'.format(k+1))
plt.show()

## By curiosity : insted of coloring by survival, color by sex. It seems that the "female" group matches a lot the "survived" group.
#fig = plt.figure(num='Plot of "male" and "female" on the first 3 PC')
#ax = fig.add_subplot(111, projection='3d')
#sex = X[:,3]
#i = 0
#j = 1
#k = 2
#for c in range(2): #2 because it is the number of different values of "sex", and we can guess them using only the column 3
#    class_mask = sex==c
#    ax.scatter(Z[class_mask,i], Z[class_mask,j],Z[class_mask,k], 'o', alpha=.5)
#plt.legend(['Male', 'Female'])
#ax.set_xlabel('PC{0}'.format(i+1))
#ax.set_ylabel('PC{0}'.format(j+1))
#ax.set_zlabel('PC{0}'.format(k+1))
#plt.show()

# Summary Statistics
# Compute values
# For Age
mean_Age = X[:,5].mean()
std_Age = X[:,5].std(ddof=1)
median_Age = np.median(X[:,5])
range_Age = X[:,5].max()-X[:,5].min()

# Display results
# print('Vector:',X[:,5])
print('Mean Age:',mean_Age)
print('Standard Deviation Age:',std_Age)
print('Median Age:',median_Age)
print('Range Age:',range_Age)

# For SibSp
mean_SibSp = X[:,6].mean()
std_SibSp = X[:,6].std(ddof=1)
median_SibSp = np.median(X[:,6])
range_SibSp = X[:,6].max()-X[:,6].min()

# Display results
# print('Vector:',X[:,6])
print('Mean SibSp:',mean_SibSp)
print('Standard Deviation SibSp:',std_SibSp)
print('Median SibSp:',median_SibSp)
print('Range SibSp:',range_SibSp)

# For Parch
mean_Parch = X[:,7].mean()
std_Parch = X[:,7].std(ddof=1)
median_Parch = np.median(X[:,7])
range_Parch = X[:,7].max()-X[:,7].min()

# Display results
# print('Vector:',X[:,7])
print('Mean Parch:',mean_Parch)
print('Standard Deviation Parch:',std_Parch)
print('Median Parch:',median_Parch)
print('Range Parch:',range_Parch)

# For Fare
mean_Fare = X[:,8].mean()
std_Fare = X[:,8].std(ddof=1)
median_Fare = np.median(X[:,8])
range_Fare = X[:,8].max()-X[:,8].min()

# Display results
# print('Vector:',X[:,8])
print('Mean Fare:',mean_Fare)
print('Standard Deviation Fare:',std_Fare)
print('Median Fare:',median_Fare)
print('Range Fare:',range_Fare)

# Statistics Pclass
# Count how many 1's are in 1st class column, ...
unique, counts_1st_class = np.unique(X[:,0], return_counts=True)
unique, counts_2nd_class = np.unique(X[:,1], return_counts=True)
unique, counts_3rd_class = np.unique(X[:,2], return_counts=True)
print('1st class passengers: ',counts_1st_class[1]);
print('2nd class pasengers: ',counts_2nd_class[1]);
print('3rd class passengers: ',counts_3rd_class[1]);

# Assign count vectors
names_1 = ['1st', '2nd', '3rd']
values_1 = [counts_1st_class[1], counts_2nd_class[1], counts_3rd_class[1]]


# Statistics Sex
# Count how many 1's are in each gender column, ...
unique, counts_male = np.unique(X[:,3], return_counts=True)
unique, counts_female = np.unique(X[:,4], return_counts=True)
print('Males: ',counts_male[1]);
print('Females: ',counts_female[1]);

# Assign count vectors
names_2 = ['male', 'female']
values_2 = [counts_male[1], counts_female[1]]

# Statistics Embarked
# Count how many 1's are in 1st class column, ...
unique, counts_c = np.unique(X[:,9], return_counts=True)
unique, counts_q = np.unique(X[:,10], return_counts=True)
unique, counts_s = np.unique(X[:,11], return_counts=True)
print('Embarked in Chesbrough: ',counts_c[1]);
print('Embarked in Queenstown: ',counts_q[1]);
print('Embarked in Southampton: ',counts_s[1]);

# Assign count vectors
names_3 = ['Cherbourg', 'Queenstown','Southampton']
values_3 = [counts_c[1], counts_q[1],counts_s[1]]

# Plot all
plt.figure(figsize=(13, 3), num='Categorical attributes')
plt.subplot(131)
plt.bar(names_1, values_1)
plt.xlabel('Passenger classes')
plt.ylabel('Number of passengers')

plt.subplot(132)
plt.bar(names_2, values_2)
plt.xlabel('Passenger gender')
plt.ylabel('Number of passengers')

plt.subplot(133)
plt.bar(names_3, values_3)
plt.xlabel('Port of embarkment')
plt.ylabel('Number of passengers')

plt.suptitle('Categorical Attributes')
plt.show()

# Boxplots
plt.figure(figsize=(13, 3), num='Boxplots')
plt.subplot(141)
boxplot(X[:,5])
ylabel('Years')
title('Boxplot of Age')

plt.subplot(142)
boxplot(X[:,6])
ylabel('Number of siblings')
title('Boxplot of SibSp')

plt.subplot(143)
boxplot(X[:,7])
ylabel('Number of parents/children')
title('Boxplot of Parch')

plt.subplot(144)
boxplot(X[:,8])
ylabel('Britain’s pre-decimalised currency')
title('Boxplot of Fare')
show()

# Normal Distribution Plot
# For Age
plt.figure(figsize=(13, 3))
plt.subplot(141)
stats.probplot(X[:,5], dist="norm", plot=pylab)
title('Probability plot of Age')

# For SibSp
plt.subplot(142)
stats.probplot(X[:,6], dist="norm", plot=pylab)
title('Probability plot of SibSp')

# For Parch
plt.subplot(143)
stats.probplot(X[:,7], dist="norm", plot=pylab)
title('Probability plot of Parch')

# For fare
plt.subplot(144)
stats.probplot(X[:,8], dist="norm", plot=pylab)
title('Probability plot of Fare')

pylab.show()

