import numpy as np

class PCA(object):
    def __init__(self):
        pass
    
    def compute_svd(self):
        """
        Computes the covariance matrix, eigenvalues, eigenvectors and singular values 
        using the numpy linear algebra library.

        Compute
        -------
        U : array_like
            The eigenvectors, representing the computed principal components
            of X. U has dimensions (n x n) where each column is a single 
            principal component.

        S : array_like
            A vector of size n, contaning the singular values for each
            principal component. Note this is the diagonal of the matrix we 
            mentioned in class.
        
        Sigma : array_like
            The covariance matrix of the dataset.

        """
        
        m = self.X.shape[0]

        self.Sigma = (1/m) * np.dot(self.X.T, self.X) 

        self.U, self.S, _ = np.linalg.svd(self.Sigma)

        return
    
    def project_data(self):
        """
        Computes the reduced data representation when projecting only 
        on to the top K eigenvectors.

        Returns
        -------
        Z : array_like
            The projects of the dataset onto the top K eigenvectors. 
            This will be a matrix of shape (m x k).

        """
        
        Ureduce = self.U[:, :self.K]
        Z = np.dot(self.X, Ureduce)
        
        return Z
    
    def fit(self, X, K=2):
        """
        Run principal component analysis. It is important to normalize 
        your dataset before running fit().

        Parameters
        ----------
        X : array_like
            The input dataset of shape (m x n). The dataset is assumed to be 
            normalized.

        K : int, default=2
            Number of dimensions to project onto. Must be smaller than n, number of features.
        """

        n = X.shape[1]
        
        self.X = X
        self.K = K
        
        assert (self.K < n)
        
        self.compute_svd()
        pca = self.project_data()
        return pca
    
    def variance(self):
        """ Calculates the amount of variance retained by the pca for K cluster"""
        
        var = np.sum(self.S[:self.K])/np.sum(self.S)
        print(f"{var*100:.4} % of variance retained for {self.K} pca clusters")
        
        return var
        
        