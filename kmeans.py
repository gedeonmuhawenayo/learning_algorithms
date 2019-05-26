import numpy as np

class KMeans(object):
    """ Clustering algorithms that groupd data into K number of clusters using distance metric"""
    
    def __init__(self, k=3, max_iters=10):
        self.K = k
        self.max_iters = max_iters
    
    def initialize_centroids(self):
        """
        Initializes K centroids by randomly selecting K examples as staring centroids.

        Returns
        -------
        centroids : array_like, size (K x n)
            Centroids of the clusters.

        """
        m = self.X.shape[0]

        # Randomly reorder the indices of examples
        randidx = np.random.permutation(m)
        # Take the first K examples as centroids
        self.centroids = self.X[randidx[:self.K], :]

        return self.centroids

    def assign_cluster(self):
        """
        Computes the centroid memberships for every example.

        Parameters
        ----------
        X : array_like
            The dataset of size (m, n) where each row is a single example. 
            That is, we have m examples each of n dimensions.

        centroids : array_like
            The k-means centroids of size (K, n). K is the number
            of clusters, and n is the the data dimension.

        Returns
        -------
        idx : array_like
            A vector of size (m, ) which holds the centroids assignment for each
            example (row) in the dataset X.
            
        """

        # Initialize
        m = self.X.shape[0] # number of examples
        idx = np.zeros(m, dtype=int)

        for i in range(m):
            row = self.X[i]
            idx[i] = np.argmin(np.sum((row - self.centroids)**2, axis=1))
        self.idx = idx
        
        return
    
    def update_centroids(self):
        """
        computes the means of the data points ssigned to each centroid and update centroid with mean accordingly.

        """

        for i in range(self.K):
            self.centroids[i] = np.mean(self.X[self.idx==i], axis=0)

        return
    
    
    def fit(self, X):
        """ Run the Kmeans algorithm on X examples
        Parameters
        ----------
        X : array_like
            The datset where each row is a single data point. That is, it 
            is a matrix of size (m, n) where there are m datapoints each
            having n dimensions.

        """
        if X.ndim == 1: 
            # promote array to 2 dimension if array is a vector
            X = X[:, None]
        self.X = X
        
        self.initialize_centroids()
        for _ in range(self.max_iters):
            self.assign_cluster()
            self.update_centroids()
        print(f"{self.K} centroids assigned for {X.shape[0]} examples")
        
        return
    
    def cost(self):
        """ Calculates the cost, C of the kmeans algorithm."""
        C = 0
        m = self.X.shape[0]
        sum_of_errors = 0.0
        
        for i in range(self.K):     
            sum_of_errors = sum_of_errors + np.sum((self.X[self.idx==i] - self.centroids[i])**2, axis=None)

        C = (1/m)*sum_of_errors
        
        return C
    
    def clusters(self):
        """ Returns the clusters assigned to each example"""
        return self.idx
    
    def show_centroids(self):
        """ Returns a dictionary of the centroid and cluster label"""
        centroid_dict = dict()
        for i, centroid in enumerate(self.centroids):
            centroid_dict[i] = list(centroid) 
        return centroid_dict
    