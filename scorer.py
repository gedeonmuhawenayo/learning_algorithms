import numpy as np

class Scorer(object):
    """ Implement different scoring metrics for learning algorithms """
    
    def __init__(self):
        pass
    
    def accuracy_(self, y, y_hat):
        return np.mean(y == y_hat)
    
    def rmse_(self, y, y_hat):
        """Calculates the root mean square error"""
        return np.sqrt(self.mse_(y, y_hat))
    
    def mse_(self, y, y_hat):
        """Calculates the mean squared error"""
        m = y.size
        mse_ = (1/m)*np.sum((y_hat-y)**2)
        return mse_