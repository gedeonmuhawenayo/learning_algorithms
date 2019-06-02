import numpy as np

class NeuralNetworkClassifier(object):
    """
    Neural Network Classification Algorithm
    
    Parameters
    -----------
    hidden_layers : array_like
        Array of activations in each hidden layer.
        e.g [10, 5] implies 10 and 5 activations in the 
        second and third layers respectively.
    Options
    ---------   
    alpha_ : float, default=0.001
        The learning rate for gradient descent.
        
    max_iters : integer, default=1000
        Maximum number of iterations to run gradient descent.

    tolerance_ : float, default=0.0001
        Difference between previous cost and current cost to 
        consider befor halting gradient descent. 
        
    """
    def __init__(self, hidden_layers, 
                 alpha_=0.001,
                 max_iters=1000, 
                 tolerance_=0.0001, 
                 ):
        self.hidden_layers = hidden_layers
        self.max_iters = max_iters
        self.tolerance_ = tolerance_
        self.alpha_ = alpha_
        print(f"NeuralNetworkClassifier(hidden_layers={self.hidden_layers}, alpha_={self.alpha_}, max_iters={self.max_iters}, tolerance_={self.tolerance_})")
        
    def __sigmoid(self, z):
        """ Compute sigmoid function given the input z """
        
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
            The weight initialiatized to random values.
            
        """

        # Randomly initialize the weights to small values
        W = np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init

        return W
    
    def forward_prop(self):
        """Forward propagates the weights and computes the cost
        
        Returns
        --------
        C : float
            The cost function of the predictions after the forward pass
        """
        
        # compute z and activation, a for each layer
        self.z = list()
        self.a = list()
        
        z_first = np.dot(self.X, self.w[0].T)
        a_first = np.concatenate([np.ones((self.num_examples,1)), self.__sigmoid(z_first)], axis=1)
        self.z.append(z_first)
        self.a.append(a_first)
        for i in range(self.num_layers - 3):
            a_i = self.a[i]
            w_i = self.w[i+1]
            z_new = np.dot(a_i, w_i.T)
            m = z_new.shape[0]
            a_new = np.concatenate([np.ones((m,1)), self.__sigmoid(z_new)], axis=1)
            self.z.append(z_new)
            self.a.append(a_new)
        z_last = np.dot(self.a[-1], self.w[-1].T)
        a_last = self.__sigmoid(z_last)
        self.z.append(z_last)
        self.a.append(a_last)
        y_hat = a_last
        
        #compute cost with normalization
        C = (1/self.num_examples)*np.sum(-1*self.y*np.log(y_hat)-(1-self.y)*np.log(1-y_hat))
        for j in range(len(self.w)):
            w_j = self.w[j][:,1:]
            C = C + (self.lambda_/(2*self.num_examples))*(np.sum(w_j**2))
        
        return C
                                          
    def backward_prop(self):
        """ Performs back propagation by taking the gradient of cost, C wrt weight, w"""
        
        delta = list()
        delta_i = self.a[-1] - self.y                            
        delta.append(delta_i)
        for i in range(self.num_layers - 2):
            index = self.num_layers-i-2
            delta_i = np.dot(delta_i, self.w[index])[:,1:]*self.__sigmoidGradient(self.z[index-1])
            delta.insert(0, delta_i)
            delta_i.shape
        self.delta = delta
                                          
        # we don't want to regularize the bias terms
        self.w_unbiased = self.w[:]                        
        for i in range(len(self.w)):                              
            self.w_unbiased[i][:, 0] = 0

        #Regularized gradients
        w_grad_1 = (1/self.num_examples)*(np.dot(self.delta[0].T, self.X)) \
                    + (self.lambda_/self.num_examples)*self.w_unbiased[0]
        self.w[0] = self.w[0] - self.alpha_*w_grad_1
        for i in range(self.num_layers - 2):
            w_grad_i = (1/self.num_examples)*(np.dot(self.delta[i+1].T, self.a[i])) \
                        + (self.lambda_/self.num_examples)*self.w_unbiased[i+1]
            # update weights
            self.w[i+1] = self.w[i+1] - self.alpha_*w_grad_i

        return

    def fit(self, X, y, lambda_=0.0):
        """ Fits the model using neural network. Sigmoid function used for non-linearity
        Parameters
        ----------
        X : array_like
            The input dataset of shape (m x n). m is the number of 
            examples, and n is the number of features.

        y : array_like
            The data labels. A vector of shape (m, ).

        lambda_ : float, optional
            The logistic regularization parameter.

        Returns
        -------
        w : array_like
            The array of weights of shape (m x n).
        cpi: array_like
            Cost per iteration. The cost calculated every 10 iteration of gradient descent.
        """
        
        # promote array to 2 dimension if array is a vector
        if X.ndim == 1:     
            X = X[:, None]
        m, n = X.shape
        self.num_examples = m
        self.X = np.concatenate([np.ones((self.num_examples, 1)), X], axis=1)
        
        self.class_values = np.unique(y) # array of classes
        self.num_labels = len(self.class_values)
        self.lambda_ = lambda_
        
        if self.num_labels > 2:
            # Turn y into one-hot-labels for multi-class case
            y_encode = np.zeros((self.num_examples, self.num_labels)) #initialize
            y_encode[range(self.num_examples), y] = 1 # used numpy advanced indexing
            self.y = y_encode
        else:
            self.y = y[:, None]
            
        # Define layer sizes
        self.input_size = n
        if self.num_labels == 2:
            self.output_size = 1
        else:
            self.output_size = self.num_labels
        # self.hidden_size = len(self.hidden_layers)
        
        self.layers = self.hidden_layers
        self.layers.insert(0, self.input_size) # add first layer
        self.layers.append(self.output_size) # add last layer
        print("NN layers:", self.layers)
        
        # Initialize weights, w
        self.num_layers = len(self.layers)
        w = list()
        for i in range(self.num_layers-1):
            w_i = self.weights_init(self.layers[i], self.layers[i+1])
            w.append(w_i)
        self.w = np.array(w)
        
        self.cost = list()
        
        for i in range(self.max_iters):
            cost_i = self.forward_prop()
            self.cost.append(cost_i)
            self.backward_prop()
            
            if not np.remainder(i, 10):
                #Display cost for every 10 iterations
                print(f"Cost for {i}th iteration - {cost_i}")
           
            if i > 0:
                current_cost = cost_i
                previous_cost = self.cost[-2]                 
                if np.abs(current_cost-previous_cost) < self.tolerance_:
                    break  
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
                                          
        prediction = np.zeros(Xval.shape[0])

        # Add ones to the X data matrix
        Xval = np.concatenate([np.ones((m, 1)), Xval], axis=1) 
        
        y_hat = self.predict_prob(Xval)
        
        prediction = np.argmax(y_hat, axis=1)                            

        return prediction
    
    def predict_prob(self, Xval):
        """Returns the matrix of probabilities of the classes"""
        m = Xval.shape[0]

        a_i = self.__sigmoid(np.dot(Xval, self.w[0].T))
        a_i = np.concatenate([np.ones((m,1)), a_i], axis=1)
        for i in range(self.num_layers - 3):
            a_i = self.__sigmoid(np.dot(a_i, self.w[i+1].T))
            a_size = a_i.shape[0]
            a_i = np.concatenate([np.ones((a_size,1)), a_i], axis=1)    
        a_last = self.__sigmoid(np.dot(a_i, self.w[-1].T))
        self.a_last = a_last

        return self.a_last
    
    def classes_(self):
        """Produces the classes and corresponding mapped integer values
        
        Returns
        ---------
        clx: dictionary
        """
        
        class_dict = dict()
        for i, class_ in enumerate(self.class_values):
            class_dict[i] = class_
        return class_dict