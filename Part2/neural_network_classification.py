import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import torch
from sklearn import model_selection
from toolbox_02450 import train_neural_net, draw_neural_net
from scipy import stats
from updateData_part2 import *


def neural_network_classification(X,y,X_test,y_test):
    # Normalize data
    X = stats.zscore(X);
    N, M = X.shape    
    # Parameters for neural network classifier
    hidden_units_ar = [1,2,4,8,16]    # number of hidden units
    n_replicates = 1        # number of networks trained in each k-fold
    max_iter = 10000         # stop criterion 2 (max epochs in training)
    
    # K-fold crossvalidation
    K = 2                   # only 2  folds to speed up
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
                                torch.nn.Linear(M, hidden_units_ar[h]), #M features to H hiden units
                                torch.nn.Tanh(),   # 1st transfer function,
                                torch.nn.Linear(hidden_units_ar[h], 1), # H hidden units to 1 output neuron
                                torch.nn.Sigmoid() # final tranfer function
                                )
            loss_fn = torch.nn.BCELoss()
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
            y_sigmoid = net(X_test)
            y_test_est = y_sigmoid>.5
            
            # Determine errors and errors
            e = y_test_est != y_test
            error_rate = (sum(e).type(torch.float)/len(y_test)).data.numpy()
            errors[h][k] = error_rate # store error rate for current h 
            print("Hidden Unit:",h,"Error:",error_rate)    
        
        
    mean_errors = np.mean(errors,axis=1)
    h_opt_ind = np.argmin(mean_errors)
    k_opt_ind = np.argmin(errors[h_opt_ind])
    
    print("Optimal Hidden Unit:",h_opt_ind,"Mean Error:",mean_errors[h_opt_ind])
    # Return the best NN for the 
    opt_NN = neural_networks[k_opt_ind][h_opt_ind] 
    print("Optimal neural network of the optimal hidden unit count:",opt_NN)
    
    y_sigmoid = opt_NN(X_test)
    y_test_est = y_sigmoid>.5   
    
    # Determine errors and errors on test data.
    e = y_test_est != y_test
    error_rate = (sum(e).type(torch.float)/len(y_test)).data.numpy()
    return error_rate, h_opt_ind

