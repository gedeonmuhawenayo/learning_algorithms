import numpy as np

class LinearSVMClassifier(object):
    """ Creates a large margin classifier. Performs both binary and multi-class linear classification.
    
    Options
    ---------
    alpha_ : float, default=0.0001
        The learning rate for gradient descent.
    
    max_iters : integer, default=100
        Maximum number of iterations to run gradient descent.

    tolerance_ : float, default=0.0001
        Difference between previous cost and current cost to consider befor halting gradient descent. 
        
    """
    
    def __init__(self, alpha_=0.1, max_iters=100, tolerance_=0.0001):
        self.alpha_ = alpha_
        self.max_iters = 100
        self.tolerance_ = tolerance_
        print(f"LogisticReg(alpha_={self.alpha_}, max_iters={self.max_iters}, tolerance_={self.tolerance_})")
    
    def cost(self):
        """Computes the cost of using weights, w as the parameter for SVM classifier.
        
        Returns
        -------
        C : float
            The computed value for the cost function.
        """
        
        y_hat = np.dot(self.X, self.w.T)
        
        C = self.gamma_*(np.sum(self.y*np.maximum(0, -1*y_hat+1)+(1-self.y)*np.maximum(0, y_hat+1)))
        
        return C
        
    def gradient_descent(self):
        """ Performs SVM gradient descent w.r.t the weights for num_iter number of iterations

        Returns
        -------
        
        grad : array_like
            A vector of shape (n, ) which is the gradient of the cost
            function with respect to weights w at the current values of w.
            
        cost_per_iter : list
            A python list for the values of the cost function after some iteration.

        """
        
        n = self.X.shape[1]
        
        # Initialize weights
        if self.num_labels > 2:
            self.w = np.zeros((self.num_labels, n))
        else:
            self.w = np.zeros((1, n))
            
        grad = np.zeros(self.w.shape)
        cost_per_iter = dict()

        # convert labels to ints if their type is bool
        if self.y.dtype == bool:
            self.y = self.y.astype(int)

        for i in range(self.max_iters):
            y_hat = np.dot(self.X, self.w.T)
            w_ = self.w[:,:]
            w_[:, 0] = 0   # because we don't add anything for bias column

            cost_1 = np.maximum(0, -1*y_hat+1)
            cost_0 = np.maximum(0, y_hat+1)
            
            grad = self.gamma_*np.sum((self.y*((cost_1!=0).astype(int))
                    +(1-self.y)*((cost_0!=0).astype(int))))
            grad = grad + w_
            
            self.w = self.w - self.alpha_*grad
            
            # save the cost in dictionary for every iteration
            cost_per_iter[i] = self.cost()
            if not np.remainder(i, 10):
                #Display cost for every 10 iterations
                print(f"Cost for {i}th iteration - {cost_per_iter[i]}")
                
            if i > 0:
                #check tolerance level of cost to stop gradient descent irrespective of num_iters
                current_cost = cost_per_iter[i]
                previous_cost = cost_per_iter[i-1]
                if np.abs(previous_cost-current_cost) <= self.tolerance_:
                    break
        return cost_per_iter

    def fit(self, X, y, gamma_=100.0, num_iters=100):
        """
        Trains an SVM to obtain the support vectors.

        Parameters
        ----------
        X : array_like
            The input dataset of shape (m x n). m is the number of 
            data points, and n is the number of features.

        y : array_like
            The data labels. A vector of shape (m, ).

        gamma_ : float. default = 1.0
            The regularization parameter.
        
        num_iters : integer, default=100

        Returns
        -------
        w : array_like
            The array of weights of shape (m x n).
        cpi: array_like
            Cost per iteration. The cost calculated every 10 iteration of gradient descent.
        """
        m = y.size # number of examples
        
        if X.ndim == 1: 
            # promote array to 2 dimension if array is a vector
            X = X[:, None]
        self.X = np.concatenate([np.ones((m, 1)), X], axis=1)
        
        # Initialize some useful variables
        self.class_values = np.unique(y)
        self.num_labels = len(self.class_values)
        self.gamma_ = gamma_
        
        # Turn y into one-hot-labels if number of classes is greater than 2
        if self.num_labels > 2:
            y_encode = np.zeros((m, self.num_labels))
            y_encode[range(m), y] = 1 #numpy advanced indexing
            self.y = y_encode
        else:
            self.y = y[:, None]
            
        cost_per_iter = self.gradient_descent()
        
        return {'w': self.w, 'cost_per_iter': cost_per_iter}
    
    def predict(self, X, threshold=0.5):
        """ Find approximate values of target variable, y_hat, using learned model weights, w
    
        Parameters
        ----------
        X : array_like
            The dataset of shape (m x n).
            
        Options
        ---------
        threshold : float, default=0.5
            Probability threshold for positive and negative classes for binary classification.

        Returns
        -------
        y_hat : array_like
            The array of predicted classes for each example, shape (m x 1).
        """
        self.threshold = threshold
        y_proba = self.predict_prob(X)
        
        if self.num_labels > 2:
            # Multiclass classification
            y_hat = np.argmax(y_proba, axis=1)
        elif self.num_labels == 2:
            # Binary classification
            y_hat = (y_proba >= self.threshold).astype(int)
        return y_hat
        
    def predict_prob(self, X):
        """Returns the matrix of probabilities of the classes instead of the predicted class"""
        
        m = X.shape[0]
        if X.ndim == 1: 
            # promote array to 2 dimension if array is a vector
            X = X[:, None]
        # Add ones to the X data matrix for the bias term
        Xval = np.concatenate([np.ones((m, 1)), X], axis=1)
        
        y_prob = np.dot(Xval, self.w.T)
        return y_prob
    
    def classes_(self):
        """Produces the classes and corresponding integer values
        
        Returns
        ---------
        clx: dictionary
        """
        
        clx = dict()
        for i, class_ in enumerate(self.class_values):
            clx[i] = class_
        return clx