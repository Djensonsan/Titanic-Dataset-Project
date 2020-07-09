# Imports
import numpy as np
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
from sklearn import model_selection
from toolbox_02450 import rlr_validate
from updateData import *
from neural_network import neural_network
from sklearn.preprocessing import StandardScaler
sc = StandardScaler() 
import scipy.stats as stats
from sklearn import model_selection
import torch
from updateData import *
from toolbox_02450 import *
from baselines import *

#Update attribute names to correspond to one-out-of-K encoding
attributeNames = ['survived', '1st class', '2nd class', '3rd class', 'male', 'age', 'sibsp', 'parch', 'S', 'Q', 'C']
M = M+1

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10 #we need 10 folds
CV = model_selection.KFold(K, shuffle=True)

# Values of lambda
lambdas = np.power(1.2,range(10,50))

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
    test_error, h_opt = neural_network(X_train,y_train,X_test,X_train)
    test_errors_neural[k][0]=h_opt
    test_errors_neural[k][1]=test_error
    print("Outer fold:",k,"Hidden Unit Index:",h_opt,"Error_test:",test_error)
    
    ## REGRESSION
    # extract training and test set for current CV fold
    X_temp = np.concatenate((np.ones((X.shape[0],1)),X),1) 

    X_train = X_temp[train_index]
    y_train = y[train_index]
    X_test = X_temp[test_index]
    y_test = y[test_index]
    internal_cross_validation = 10   
    
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)

    # Standardize outer fold based on training set, and save the mean and standard
    # deviations since they're part of the model (they would be needed for
    # making new predictions)
    mu[k, :] = np.mean(X_train[:, 1:], 0)
    sigma[k, :] = np.std(X_train[:, 1:], 0)
    
    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :] ) / sigma[k, :] 
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :] ) / sigma[k, :] 
    
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    
    # Compute mean squared error without using the input data at all
    Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum(axis=0)/y_train.shape[0]
    Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum(axis=0)/y_test.shape[0] 

    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0,0] = 0 # Do no regularize the bias term
    w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    # Compute mean squared error with regularization with optimal lambda
    Error_train_rlr[k] = np.square(y_train-X_train @ w_rlr[:,k]).sum(axis=0)/y_train.shape[0]
    Error_test_rlr[k] = np.square(y_test-X_test @ w_rlr[:,k]).sum(axis=0)/y_test.shape[0]
    print('Outer fold:',k,'Optimal lambda:',opt_lambda,'Error_test:',Error_test_rlr[k])
    test_errors_regression[k][0]=opt_lambda
    test_errors_regression[k][1]=Error_test_rlr[k]

    # Estimate weights for unregularized linear regression, on entire training set
    w_noreg[:,k] = np.linalg.solve(XtX,Xty).squeeze()
    # Compute mean squared error without regularization
    Error_train[k] = np.square(y_train-X_train @ w_noreg[:,k]).sum(axis=0)/y_train.shape[0]
    Error_test[k] = np.square(y_test-X_test @ w_noreg[:,k]).sum(axis=0)/y_test.shape[0]

    # Display the results for the last cross-validation fold
    if k == K-1:
        figure(k, figsize=(12,8))
        subplot(1,2,1)
        semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
        xlabel('Regularization factor')
        ylabel('Mean Coefficient Values')
        grid()

        subplot(1,2,2)
        title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
        loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
        xlabel('Regularization factor')
        ylabel('Squared error (crossvalidation)')
        legend(['Train error','Validation error'])
        grid()
    
    
    ##Baseline model
    X_train = X[train_index] 
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    
    test_errors_baseline[k] = baseline_regression(X_train, X_test, y_train, y_test)

    k+=1

show()
# Display results
print('Linear regression without feature selection:')
print('- Training error: {0}'.format(Error_train.mean()))
print('- Test error:     {0}'.format(Error_test.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
print('Regularized linear regression:')
print('- Training error: {0}'.format(Error_train_rlr.mean()))
print('- Test error:     {0}'.format(Error_test_rlr.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_rlr.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test_rlr.sum())/Error_test_nofeatures.sum()))

print('Weights in last fold:')
attributeNames = ['offset', 'survived', '1st class', '2nd class', '3rd class', 'male', 'age', 'sibsp', 'parch', 'S', 'Q', 'C']
for m in range(M):
    print('{:>15} {:>15}'.format(attributeNames[m], np.round(w_rlr[m,-1],2)))

#display results
hidden_units_ar = [1,2,4,8,16] #from the code of the neural network
print('Outer fold              ANN                           Linear regression                           Baseline' )
print('i             h              E                    lambda                       E                      E') 
for i in range(10):
    print(i+1 , '          ',hidden_units_ar[int(test_errors_neural[2,0])], '   ',test_errors_neural[i,1], '     ', test_errors_regression[i,0], '     ', test_errors_regression[i,1], '      ', test_errors_baseline[i])
print('Ran comparison')

