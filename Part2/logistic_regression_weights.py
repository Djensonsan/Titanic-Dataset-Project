#This is just to display the weights found by the logistic regression for the best lambda, since the logistic regression code prints the weights of the last fold
# Imports
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from toolbox_02450 import rocplot, confmatplot
from updateData_part2 import *
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

font_size = 15
plt.rcParams.update({'font.size': font_size})

y=y.squeeze()
# Create crossvalidation partition for evaluation
# using stratification and 95 pct. split between training and test 
K = 10
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.95, stratify=y)

# Standardize the training and set set based on training set mean and std
mu = np.mean(X_train, 0)
sigma = np.std(X_train, 0)

X_train = (X_train - mu) / sigma
X_test = (X_test - mu) / sigma


# Fit regularized logistic regression model to training data
best_lambda = 2.29
train_error_rate = 0
test_error_rate = 0
coefficient_norm = 0

mdl = LogisticRegression(penalty='l2', C=1/best_lambda)

mdl.fit(X_train, y_train)

y_train_est = mdl.predict(X_train).T
y_test_est = mdl.predict(X_test).T

train_error_rate= np.sum(y_train_est != y_train) / len(y_train)
test_error_rate= np.sum(y_test_est != y_test) / len(y_test)

w_est = mdl.coef_[0] 
print(w_est)
coefficient_norm = np.sqrt(np.sum(w_est**2))



print('Weights:')
for m in range(M):
    print('{:>15} {:>15}'.format(attributeNames[m], np.round(w_est[m],2)))


print('Ran classification - logistic regression weights')
