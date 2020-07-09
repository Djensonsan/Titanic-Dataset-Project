"""
Created on Sun Nov 10 18:32:56 2019

@author: jenselin
"""
# Imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
sc = StandardScaler() 
import scipy.stats as stats
from sklearn import model_selection
import torch
plt.rcParams.update({'font.size': 12})
from updateData import *
from toolbox_02450 import *

# Function should return best neural network of the optimal hidden unit count.
# Input to function is the 
def neural_network(X,y,X_test,y_test):

    y = y[:,np.newaxis]
    # Normalize data
    X = stats.zscore(X);
    
    #adapt the type of attributeNames
    attributeNames = ['survived', '1st class', '2nd class', '3rd class', 'male', 'age', 'sibsp', 'parch', 'S', 'Q', 'C']
    attributeNames=[(np.str_)(i) for i in attributeNames] 
                            
    N, M = X.shape
    
    # Parameters for neural network classifier
    hidden_units_ar = [1,2,4,8,16]
    n_replicates = 1        # number of networks trained in each k-fold
    max_iter = 10000         
    
    # K-fold crossvalidation
    K = 2 # 2 to speed up the process
    CV = model_selection.KFold(K, shuffle=True)
    
    errors = np.zeros((len(hidden_units_ar),K)) # make a list for storing generalizaition error in each loop
    neural_networks = [[0 for x in range(len(hidden_units_ar))] for y in range(K)]
    for (k, (train_index, test_index)) in enumerate(CV.split(X,y)): 
        print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))    
        # Extract training and test set for current CV fold, convert to tensors
        X_train = torch.tensor(X[train_index,:], dtype=torch.float)
        y_train = torch.tensor(y[train_index], dtype=torch.float)
        X_test = torch.tensor(X[test_index,:], dtype=torch.float)
        y_test = torch.tensor(y[test_index], dtype=torch.uint8)
        
        
        for h in range(0,len(hidden_units_ar)):
            print('\nHidden Unit: {0}'.format(h)) 
            # Define the model
            model = lambda: torch.nn.Sequential(
                                torch.nn.Linear(M, hidden_units_ar[h]), #M features to n_hidden_units
                                torch.nn.Tanh(),   # 1st transfer function,
                                torch.nn.Linear(hidden_units_ar[h], 1), # n_hidden_units to 1 output neuron
                                # no final tranfer function, i.e. "linear output"
                                )
            loss_fn = torch.nn.MSELoss() 
            # Train the net on training data
            net, final_loss, learning_curve = train_neural_net(model,
                                                               loss_fn,
                                                               X=X_train,
                                                               y=y_train,
                                                               n_replicates=n_replicates,
                                                               max_iter=max_iter)
            # Save the net
            neural_networks[k][h]=net
            # Determine estimated class labels for test set
            y_test_est = net(X_test)
            
            # Determine errors and errors
            se = (y_test_est.float()-y_test.float())**2 # squared error
            mse = (sum(se).type(torch.float)/len(y_test)).data.numpy() #mean
            errors[h][k] = mse # store error rate for current h 
            print("Hidden Unit:",h,"Error:",mse)
    
    mean_errors = np.mean(errors,axis=1)
    h_opt_ind = np.argmin(mean_errors)
    k_opt_ind = np.argmin(errors[h_opt_ind])
    
    print("Optimal Hidden Unit:",h_opt_ind,"Mean Error:",mean_errors[h_opt_ind])
    # Return the best NN for the 
    opt_NN = neural_networks[k_opt_ind][h_opt_ind] 
    print("Optimal neural network of the optimal hidden unit count:",opt_NN)
    
    y_test_est = opt_NN(X_test)     
    # Determine errors and errors on test data.
    se = (y_test_est.float()-y_test.float())**2 # squared error
    mse = (sum(se).type(torch.float)/len(y_test)).data.numpy() #mean
    
    return mse, h_opt_ind

