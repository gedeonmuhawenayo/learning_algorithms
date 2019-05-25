import numpy as np
from scorer import Scorer

class LinearRegression(object):
    """
    Fits a dataset (X, y) using a linear model 
    
    Options
    ----------
    alpha : float, default: 0.01
        The learning rate for gradient descent.
    num_iters : float, default=1500
        Maximum iterations to run gradient descent.
    num_iters: integer, default: 1500
        Maximum number of iterations to run gradient descent. 
        If normal equation is used, it is not considered.
        
    Returns
    -------
    X_norm : array_like
        The normalized dataset of shape (m x n).
    """
    
    def __init__(self, alpha_=0.01, num_iters=1500):
        self.alpha_ = alpha_
        self.num_iters = num_iters
        print(f"LinearRegression(alpha_={self.alpha_}, num_iters={self.num_iters})")
        
    def cost(self):
        """
        Compute cost for linear regression. Computes the cost of using theta as
        the parameter for linear regression to fit the data points in X and y. 
            
        Parameters
        ----------
        X : array_like
            The dataset. Matrix with shape (m x n + 1) where m is the 
            total number of examples, and n is the number of features 
            before adding the bias term.

        y : array_like
            The functions values at each datapoint. A vector of
            shape (m, ).

        w : array_like
            The parameters for linear regression. A vector of shape (n+1,).

        lambda_ : float, optional
            The regularization parameter.

        Returns
        -------
        C : float
            The value of the cost function.
        """    
        # Initialize some useful values
        m = self.y.size # number of training examples
        C = 0

        y_hat = np.dot(self.X, self.w.T)
        C = (1/(2*m)) * np.sum((y_hat - self.y)**2)

        return C
    
    def gradient_descent(self):
        """
        Performs gradient descent to learn the weights w.
        Update weights by taking num_iters gradient steps with learning rate alpha.
    
        Returns
        -------
        cpi : list
            A python list for the values of the cost function after some iteration.
        """
        
        # Initialize some useful values
        m = self.y.size # number of training examples
        self.w = np.zeros(self.X.shape[1])
        cpi = dict()
        
        for i in range(self.num_iters):
            y_hat = np.dot(self.X, self.w.T)           
            
            w_ = self.w.copy()
            w_[0] = 0   # because we don't add anything for j = 0
            grad = (1/m) * np.dot(self.X.T, (y_hat - self.y))
            grad = grad - (self.lambda_/m)*w_
            self.w = self.w - self.alpha_*grad
            
            # save the cost C in dictionary for every Kth iteration
            if not np.remainder(i, 10):
                cpi[i] = self.cost()
        return cpi
    
    def normal_eqn(self):
        """
        Computes the closed-form solution to linear regression using the normal equations.
        w = np.linalg.inv(X.T@X)@(X.T)@y

        Parameters
        ----------
        X : array_like
            The dataset of shape (m x n+1).

        y : array_like
            The value at each data point. A vector of shape (m, ).

        Returns
        -------
        w : array_like
            Estimated linear regression weights. A vector of shape (n+1, ).
        """
        self.w = np.zeros(self.X.shape[1]) #initialize
        self.w = ((np.linalg.pinv(np.dot(self.X.T, self.X)))@self.X.T)@self.y
        
        return
    
    def fit(self, X, y, lambda_=0.0, normal=False):
        """ Fits a dataset (X, y) using a linear model 
    
        Parameters
        ----------
        X : array_like
            The dataset of shape (m x n).

        y : array_like
            A vector of shape (m, ) for the values at a given data point.
            
        normal : boolean, default : False
            Specify the optimization method to use, whether normal equation or gradient descent.
            Uses gradient descent by default
            
         Options
        ----------        
        lambda_ : float, optional
            The regularization parameter. Set to 0.0 to use normal linear regression without regularization.

        Returns
        -------
        w : array_like
            The array of weights of shape (m x n).
        cpi: array_like
            Cost per iteration. The cost calculated over each iteration of gradient descent.

        """
        
        m = y.size # number of examples
        
        if X.ndim == 1: 
            #promote array to 2 dimension if array is a vector
            X = X[:, None]
            self.X = np.concatenate([np.ones((m, 1)), X], axis=1)
        else:
            self.X = np.concatenate([np.ones((m, 1)), X], axis=1)
        self.y = y
        self.lambda_ = lambda_
        self.w = np.zeros(self.X.shape[1])
        
        if normal: 
            cpi = self.normal_eqn()
        else:
            cpi = self.gradient_descent()    
        
        return {'w': self.w, 'cpi': cpi}
    
    def predict(self, X, y):
        """ Find approximate values of target variable, y_hat, using learned model weights, w
    
        Parameters
        ----------
        X : array_like
            The dataset of shape (m x n).

        y : array_like
            A vector of shape (m, ) for the values at a given data point.

        Returns
        -------
        A dictionary of y_hat and score
        
        y_hat : array_like
            The array of predicted target for each example, shape (m x n).
        """
        m = X.shape[0]
        
        if X.ndim == 1: 
            #promote array to 2 dimension if array is a vector
            X = X[:, None]
            X = np.concatenate([np.ones((m, 1)), X], axis=1)
        else:
            X = np.concatenate([np.ones((m, 1)), X], axis=1)
        
        y_hat = np.dot(X, self.w.T)
        
        return y_hat
    
    def score(self, y, y_hat, how="rmse"):
        """ 
        Calculates score metrics for the learning algorithm.
        Parameters
        ----------
        X : array_like
            The dataset of shape (m x n).

        y : array_like
            A vector of shape (m, ) for the values at a given data point.
        
        Options
        ----------        
        type : string
            The type of metric to be used in evaluation.
            - rmse : root mean squared error
            - mse : mean squared error

        Returns
        -------
        score: float
        """
        scorer = Scorer()
        if how == "rmse":   
            score = scorer.rmse_(y, y_hat)
            print(f"rmse: {score}")
        elif how == "mse":
            score = scorer.mse_(y, y_hat)
            print(f"mse: , {score}")