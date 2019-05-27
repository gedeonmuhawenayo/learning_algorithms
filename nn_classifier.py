import numpy as np

class NeuralNetworkClassifier(object):
    """
    Neural Network Classification Algorithm
    
    Options
    ---------   
    alpha_ : float, default=0.001
        The learning rate for gradient descent.
        
    max_iters : integer, default=100
        Maximum number of iterations to run gradient descent.

    tolerance_ : float, default=0.0001
        Difference between previous cost and current cost to consider befor halting gradient descent. 
        
    """
    def __init__(self, hidden_layers, 
                 alpha_=0.001,
                 max_iters=100, 
                 tolerance_=0.0001, 
                 ):
        self.hidden_layers = hidden_layers
        self.max_iters = max_iters
        self.tolerance_ = tolerance_
        self.alpha_ = alpha_
        
    def __sigmoid(self, z):
        """ Compute sigmoid function given the input z"""
        # convert input to a numpy array
        z = np.array(z)

        # Initialize
        g = np.zeros(z.shape)
        g = 1/(1 + np.exp(-1*z))

        return g
    
    def __sigmoidGradient(self, z):
        """
        Computes the gradient of the sigmoid function evaluated at z.

        Parameters
        ----------
        z : array_like
            A vector or matrix as input to the sigmoid function. 

        Returns
        --------
        g : array_like
            Gradient of the sigmoid function. Has the same shape as z. 
        """

        gz = self.__sigmoid(z)
        g = gz*(1-gz)

        return g
    
    def weights_init(self, L_in, L_out, epsilon_init=0.12):
        """
        Randomly initialize the weights of a layer in a neural network to break symmetry while training.

        Parameters
        ----------
        L_in : int
            Number of incomming connections.

        L_out : int
            Number of outgoing connections. 

        epsilon_init : float, optional
            Range of values which the weight can take from a uniform 
            distribution.

        Returns
        -------
        W : array_like
            The weight initialiatized to random values.  Note that W should
            be set to a matrix of size(L_out, 1 + L_in) as
            the first column of W handles the "bias" terms.
            
        """

        # Randomly initialize the weights to small values
        W = np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init

        return W
    
    def forward_prop(self):
        """Forward propagates the weights and computes the cost"""
        # Setup some useful variables
        m = self.y.size
        
        z_i = np.dot(self.X, self.w[0].T)
        for i in range(self.num_layers - 2):
            a_i = np.concatenate([np.ones((m,1)), self.__sigmoid(z_i)], axis=1)
            self.z.append(z_i)
            self.a.append(a_i)
        z_l = np.dot(self.a[-1], self.z[-1].T)
        a_l = self.__sigmoid(z_l)
        
        y_hat = a_l

        if not y_encode and self.num_labels > 2:
            # Turn y into one-hot-labels if not yet done
            y_encode = np.zeros((m, self.num_labels)) #initialize
            y_encode[range(m), self.y] = 1 # numpy advanced indexing
            self.y_encode = y_encode
        
        #compute cost with normalization
        J = (1/m)*np.sum(-1*self.y_encode*np.log(y_hat)-(1-self.y_encode)*np.log(1-y_hat))
        for j in self.w:
            J = J + (self.lambda_/(2*m))*(np.sum(self.w[j][:,1:]**2))
        
        return J
                                          
    def backward_prop(self):
        m = self.y.size

        delta = list()
        delta_i = self.a[-1] - self.y_encode                             
        delta.append(delta_i)
        for i in range(self.num_layers-2):
            index = self.num_layers-i-2
            delta_i = np.dot(delta_i, self.w[index])[:,1:]*self.__sigmoidGradient(self.z[index-1])
            delta.insert(0, delta_i)
        self.delta = delta
                                          
        # we don't want to regularize the bias terms
        self.w_unbiased = self.w[:,:]                        
        for i in range(len(self.w)):                              
            self.w_unbiased[i][:, 0] = 0

        #Regularized gradients
        w_grads = list()                           
        for i in range(self.num_layers - 1):
            w_grads_i = (1/m)*(np.dot(self.delta[i].T, self.X)) + (self.lambda_/m)*self.w_unbiased[i]
            self.w[i] = self.w[i] - self.alpha_*w_grads_i

        return
    
    
    def fit(self, X, y, lambda_=0.0, num_iters=100):
        if X.ndim == 1: 
            # promote array to 2 dimension if array is a vector
            X = X[:, None]
        m, n = X.shape
        self.X = np.concatenate([np.ones((m, 1)), X], axis=1)
        
        self.class_values = np.unique(y)
        self.num_labels = len(self.class_values)
        self.lambda_ = lambda_
        self.y = y
        
        self.input_size = n
        if self.num_labels == 2:
            self.output_size = 1
        else:
            self.output_size = self.num_labels
        self.hidden_size = len(self.hidden_layers)
        
        self.layers = self.hidden_layers
        self.layers.insert(0, self.input_size) # add first layer
        self.layers.append(self.output_size) # add last layer
        print("NN layers:", self.layers)
        
        # Initialize weights, w
        self.num_layers = len(self.layers)
        w = list()
        for i in range(self.num_layers-1):
            w_i = self.weights_init(self.layers[i], self.layers[i+1])
            print(w_i.shape)
            w.append(w_i)
        self.w = w
        self.z = list()
        self.a = list()
        
        for i in range(self.max_iters):
            self.forward_prop()
            self.backward_prop()
        
        return
                                          
    def predict(self, Xval):
        """
        Predict the label of an input given a trained neural network.

        Parameters
        ----------
        X : array_like
            The examples having shape (number of examples x no. of features).

        Return 
        ------
        p : array_like
            Predictions vector containing the predicted label for each example.
            It has a length equal to the number of examples.
        """
        # If input is of one dimension, promote to 2-dimension
        if Xval.ndim == 1: 
            # promote array to 2 dimension if array is a vector
            Xval = Xval[:, None]

        # useful variables
        m = Xval.shape[0]
                                          
        p = np.zeros(Xval.shape[0])

        # Add ones to the X data matrix
        Xval = np.concatenate([np.ones((m, 1)), Xval], axis=1) 
        z = list()
        a = list()
        for i in range(self.num_layers - 1):
            z_i = np.dot(self.X, self.w[i].T)
            a_i = np.concatenate([np.ones((m,1)), self.__sigmoid(z_i)], axis=1)
            z.append(z_i)
            a.append(a_i)
        z_l = np.dot(a[-1], z[-1].T)
        a_l = self.__sigmoid(z_l)
        
                                                                                                        
        a2 = self.__sigmoid(np.dot(self.X, self.w[0].T))

        # Add ones to the a2 data matrix
        a2 = np.concatenate([np.ones((m, 1)), a2], axis=1) 

        a3 = self.__sigmoid(np.dot(a2, self.w[1].T))
        p = np.argmax(a3, axis=1)

        return p