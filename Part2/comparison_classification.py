# Imports
import numpy as np
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
from sklearn import model_selection
from toolbox_02450 import rlr_validate
from neural_network_classification import neural_network_classification
from sklearn.preprocessing import StandardScaler
sc = StandardScaler() 
import scipy.stats as stats
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import torch
from updateData_part2 import *
from toolbox_02450 import *
from baselines import *

#Update attribute names to correspond to one-out-of-K encoding
attributeNames = ['1st class', '2nd class', '3rd class', 'male', 'age', 'sibsp', 'parch','fare', 'S', 'Q', 'C']
M = M+1

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10 #we need 10 folds
CV = model_selection.KFold(K, shuffle=True)

# Values of lambda
lambda_interval = np.logspace(-8, 10, 100)

# Initialize variables
Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
Error_train_rlr = np.empty((K,1))
Error_test_rlr = np.empty((K,1))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))
w_rlr = np.empty((M,K))
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))
w_noreg = np.empty((M,K))

test_errors_neural=np.zeros((10,2))
test_errors_regression=np.zeros((10,2))
test_errors_baseline=np.zeros(10)

k=0
for train_index, test_index in CV.split(X,y):
    ## ANN
    X_train = X[train_index] 
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    
    internal_cross_validation = 10   
    test_error, h_opt = neural_network_classification(X_train,y_train,X_test,X_train)
    test_errors_neural[k][0]=h_opt
    test_errors_neural[k][1]=test_error
    print("Outer fold:",k,"Hidden Unit Index:",h_opt,"Error_test:",test_error)
    
    ## REGRESSION

    y_temp=y.squeeze()

    X_train = X[train_index] 
    y_train = y_temp[train_index]
    X_test = X[test_index]
    y_test = y_temp[test_index]
    mu = np.mean(X_train, 0)
    sigma = np.std(X_train, 0)
    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma
    
    internal_cross_validation = 10   
    
    train_error_rate = np.zeros(len(lambda_interval))
    test_error_rate = np.zeros(len(lambda_interval))
    coefficient_norm = np.zeros(len(lambda_interval))
    for ki in range(0, len(lambda_interval)):
        mdl = LogisticRegression(penalty='l2', C=1/lambda_interval[ki] )
        
        mdl.fit(X_train, y_train)
    
        y_train_est = mdl.predict(X_train).T
        y_test_est = mdl.predict(X_test).T
        
        train_error_rate[ki] = np.sum(y_train_est != y_train) / len(y_train)
        test_error_rate[ki] = np.sum(y_test_est != y_test) / len(y_test)
    
        w_est = mdl.coef_[0] 
        coefficient_norm[ki] = np.sqrt(np.sum(w_est**2))
    
    min_error = np.min(test_error_rate)
    opt_lambda_idx = np.argmin(test_error_rate)
    opt_lambda = lambda_interval[opt_lambda_idx]
    print('Outer fold:',k,'Optimal lambda:',opt_lambda,'Error_test:',test_error_rate[k])
    test_errors_regression[k][0]=opt_lambda
    test_errors_regression[k][1]=test_error_rate[k]

    
    ##Baseline model
    X_train = X[train_index] 
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    
    test_errors_baseline[k] = baseline_classification(X_train, X_test, y_train, y_test)

    k+=1


#display results
hidden_units_ar = [1,2,4,8,16] #from the code of the neural network
print('Outer fold              ANN                           Linear regression                           Baseline' )
print('i             h              E                    lambda                       E                      E') 
for i in range(10):
    print(i+1 , '          ',hidden_units_ar[int(test_errors_neural[2,0])], '   ',test_errors_neural[i,1], '     ', test_errors_regression[i,0], '     ', test_errors_regression[i,1], '      ', test_errors_baseline[i])
print('Ran comparison - classification')

