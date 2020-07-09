# Imports
import numpy as np

#Fort part 1
def baseline_regression(X, y, X_test,y_test):
    moy=y.mean()
    y_predicted = moy*np.ones(len(y))
    se = np.square(y_predicted-y_test)
    mse=np.sum(se)/len(y)
    return mse
    
    
#For part 2    
def baseline_classification(X, y, X_test,y_test):
    count_ones = np.count_nonzero(y==1)
    count_zeros = np.count_nonzero(y==0)
    if count_ones > count_zeros:
        y_predicted = np.ones(len(y))
    else :
        y_predicted = np.zeros(len(y))
    error_rate=0
    for i in range(len(y_predicted)):
        if y_predicted[i]!=y_test[i]:
            error_rate+=1
    error_rate=error_rate/len(y_predicted)
    return error_rate
    
    
