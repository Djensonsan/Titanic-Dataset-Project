#Processing the data in order to compute the regression
#The data is stored in X and the ouputs (the survived status) in y

#Imports
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import torch
from sklearn import model_selection
from toolbox_02450 import train_neural_net, draw_neural_net
from scipy import stats
import pandas as pd
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

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
y = pd.to_numeric(survived);
y = y[:,np.newaxis]
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

#Encode the one-out-of-K values
label_encoder = LabelEncoder()
onehot_encoder = OneHotEncoder(categories = 'auto',sparse=False)

# Code Pclass values
pclass = X[:,0];
onehot_pclass_encoded = onehot_encoder.fit_transform(pclass.reshape(len(pclass),1))

# Code embarked status
embarked = X[:,-1];
integer_embarked_encoded = label_encoder.fit_transform(embarked)
integer_embarked_encoded = integer_embarked_encoded.reshape(len(integer_embarked_encoded),1)
onehot_embarked_encoded = onehot_encoder.fit_transform(integer_embarked_encoded)

#replacing the columns with the encoded values
A = onehot_pclass_encoded
r = np.reshape(X[:,1], (712, 1))
A = np.hstack((A, r))
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


# Turn the data matrix in an array of floats, now that all the data has been changed to floats
X = np.asarray(X.astype(float))


attributeNames = ['1st class', '2nd class', '3rd class', 'male', 'age', 'sibsp', 'parch','fare', 'S', 'Q', 'C']

N, M = X.shape
classNames = ['died', 'survived']
C = 2