from apyori import apriori
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

###### Functions
# This is a helper function that transforms a binary matrix into transactions.
def mat2transactions(X, labels=[]):
    T = []
    for i in range(X.shape[0]):
        l = np.nonzero(X[i, :])[0].tolist()
        if labels:
            l = [labels[i] for i in l]
        T.append(l)
    return T


# This function prints the found rules and also returns a list of rules in the format:
# [(x,y), ...]
# where x -> y
def print_apriori_rules(rules):
    frules = []
    for r in rules:
        for o in r.ordered_statistics:        
            conf = o.confidence
            supp = r.support
            x = ", ".join( list( o.items_base ) )
            y = ", ".join( list( o.items_add ) )
            print("{%s} -> {%s}  (supp: %.3f, conf: %.3f)"%(x,y, supp, conf))
            frules.append( (x,y) )
    return frules

####### Prepare data and making it binary
# Load the data
filename = './train.csv'
df = pd.read_csv(filename) #panda returns a dataframe

# Deleting the columns and lines that are not interesting
df = df.drop("PassengerId", axis=1)
df = df.drop("Name", axis=1)
df = df.drop("Ticket", axis=1)
df = df.drop("Cabin", axis=1)
df = df.dropna() #gets rid of the rows containing nan values
raw_data = np.asarray(df); #converting the dataframe into a numpyarray

# Creating the data matrix X
cols= range(0,8) #select the indexes of the columns to be used
X = raw_data[:,cols] #making the data matrix X
rowSize, colSize = X.shape

# Change "male" and "female" to 1 and 0. for reference : X[row][col]
for i in range(rowSize): 
    if X[i,2]=="male":
        X[i,2]=1;
    elif X[i,2]=="female":
        X[i,2]=0;
        
# Change the Embarked status
for i in range(rowSize): 
    if X[i,-1]=="S":
        X[i,-1]=0;
    elif X[i,-1]=="Q":
        X[i,-1]=1;
    elif X[i,-1]=="C":
        X[i,-1]=2;


# Binarize Age
median_age = np.median(X[:, 3]);
for i in range(rowSize):
    if X[i, 3]<=median_age:
        X[i, 3]=0;
    else:
        X[i, 3]=1;


# Binarize SibSp
median_sibsp = np.median(X[:, 4]);
for i in range(rowSize):
    if X[i, 4]<=median_sibsp:
        X[i, 4]=0;
    else:
        X[i, 4]=1;
        
# Binarize Parch
median_parch = np.median(X[:, 5]);
for i in range(rowSize):
    if X[i, 5]<=median_parch:
        X[i, 5]=0;
    else:
        X[i, 5]=1;
        
# Binarize Fare
median_fare = np.median(X[:, 6]);
for i in range(rowSize):
    if X[i, 6]<=median_fare:
        X[i, 6]=0;
    else:
        X[i, 6]=1;
        
X = np.asarray(X.astype(float))


#Encode using one-of-K encoding
label_encoder = LabelEncoder()
onehot_encoder = OneHotEncoder(categories = 'auto',sparse=False)

# Code Survived values
survived = X[:,0];
onehot_survived_encoded = onehot_encoder.fit_transform(survived.reshape(len(survived),1))
                                                                
# Code Pclass values
pclass = X[:,1];
onehot_pclass_encoded = onehot_encoder.fit_transform(pclass.reshape(len(pclass),1))

# Code sex values
sex = X[:,2];
onehot_sex_encoded = onehot_encoder.fit_transform(sex.reshape(len(sex),1))

# Code age values
age = X[:,3];
onehot_age_encoded = onehot_encoder.fit_transform(age.reshape(len(age),1))

# Code sibsp values
sibsp = X[:,4];
onehot_sibsp_encoded = onehot_encoder.fit_transform(sibsp.reshape(len(sibsp),1))

# Code parch values
parch = X[:,5];
onehot_parch_encoded = onehot_encoder.fit_transform(parch.reshape(len(parch),1))

# Code fare values
fare = X[:,6];
onehot_fare_encoded = onehot_encoder.fit_transform(fare.reshape(len(fare),1))

# Code embarked status
embarked = X[:,-1];
integer_embarked_encoded = label_encoder.fit_transform(embarked)
integer_embarked_encoded = integer_embarked_encoded.reshape(len(integer_embarked_encoded),1)
onehot_embarked_encoded = onehot_encoder.fit_transform(integer_embarked_encoded)


#replacing the columns with the encoded values
A = onehot_survived_encoded 
A = np.hstack((A, onehot_pclass_encoded)) 
A = np.hstack((A, onehot_sex_encoded)) 
A = np.hstack((A, onehot_age_encoded)) 
A = np.hstack((A, onehot_sibsp_encoded)) 
A = np.hstack((A, onehot_parch_encoded)) 
A = np.hstack((A, onehot_fare_encoded)) 
A = np.hstack((A, onehot_embarked_encoded)) 
X=np.asarray(A.astype(float));


attributeNamesBin = ['died', 'survived', '1st class', '2nd class', '3rd class', 'female', 'male', 'age<median', 'age>median', 'no sibsp', 'sibsp', 'no parch ','parch','fare<median', 'fare>median', 'S', 'Q', 'C']
N, M = X.shape





print("X, i.e. the titanic dataset, has now been transformed into:")
print(X)
print(attributeNamesBin)


T = mat2transactions(X,labels=attributeNamesBin)
rules = apriori(T, min_support=0.08, min_confidence=0.8)
print_apriori_rules(rules)


