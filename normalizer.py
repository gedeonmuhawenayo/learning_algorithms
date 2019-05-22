import numpy as np

class Normalizer(object):
    """ Class of functions applying various scaling to the array X. 
        see https://en.wikipedia.org/wiki/Feature_scaling for scaling methods.
             
    """
    def __init__(self):
        pass
        
    def feature_centre(self, X):
        """ Subtracts the mean from each feature
        
        Parameters
        ----------
        X : array_like
            The dataset of shape (m x n).

        Returns
        -------
        X_norm : array_like
            The centered dataset of shape (m x n).
        mu: array_like
            The mean of the features (n, )
        """
        self.mu = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.X_norm = X - mu
        
        return self.X_norm, self.mu
    
    def min_max_normalizer(self, X):
        """
        Normalizes the features in X using min-max normalization. It subtracts the minimum of a feature and divides 
        by the range. Returns a normalized version of X in the range [0, 1]
 
        Parameters
        ----------
        X : array_like
            The dataset of shape (m x n).

        Returns
        -------
        X_norm : array_like
            The normalized dataset of shape (m x n).
            
        min_ : array_like
            The mean array of features.
            
        range_ : array_like
            The diffrence between the aximum and minimum values of features
        """
        self.min_ = np.min(X, axis=0)
        self.range_ = np.max(X, axis=0) - self.min_
        self.X_norm = (X - self.min_)/self.range_
        
        return self.X_norm, self.min_, self.range_
        
    def mean_normalizer(self, X):
        """
        Normalizes the features in X using  normalization. It subtracts the mean of a feature and divides 
        by the standard deviation. Returns a normalized version of X with mean 0 and standard deviation 1.
 
        Parameters
        ----------
        X : array_like
            The dataset of shape (m x n).

        Returns
        -------
        X_norm : array_like
            The normalized dataset of shape (m x n).
        mu : array_like
            The mean of the features (n, ).
        std : array_like
            The standard deviation of features (n, ).
        """
        
        self.mu = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        
        self.X_norm = (X - self.mu)/self.std
        
        return self.X_norm, self.mu, self.std