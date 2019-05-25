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
    
    def confusion_matrix(self, y, y_hat):
        true_positive = np.sum(y*y_hat)
        false_positive = np.sum((y==0)*y_hat)
        true_negative = np.sum((y==0)*(y_hat==0))
        false_negative = np.sum(y*(y_hat==0))
        
        return {"true_positive":true_positive, 
                "false_positive":false_positive,
                "true_negative":true_negative, 
                "false_negative":false_negative
               }
    
    def precision(self, y, y_hat):
        cm = self.confusion_matrix(y, y_hat)
        precision = cm["true_positive"]/(cm["true_positive"]+cm["false_positive"])
        return precision
    
    def recall(self, y, y_hat):
        cm = self.confusion_matrix(y, y_hat)
        recall = cm["true_positive"]/(cm["true_positive"]+cm["false_negative"])
        return recall
    
    def f1_score(self, y, y_hat):
        precision = self.precision(y, y_hat)
        recall = self.recall(y, y_hat)
        f1_score = (2*precision*recall)/(precision+recall)
        return f1_score