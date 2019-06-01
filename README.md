# Project Title

Learning Algorithms written from scratch using numpy.

## Getting Started

Clone the repo   
    ``` git clone https://github.com/ogunlao/learning_algorithms.git ```

or download a zipped version on your local machine.

### Prerequisites

* python >= 3.6
* numpy >= 1.14  

Python can be downloaded from <a href="https://www.python.org/">Python website</a> while information on numpy can be found on the <a href="https://www.numpy.org/">numpy official website</a>

### Contents

* Linear Regression algorithm with L2 regularization
* Logistic Regression
* Neural Networks Classifier
    * Forward Propagation
    * Back Propagation
* KMeans clustering algorithm
* Principal Component Analysis (PCA)
* Scoring class which includes Accuracy, f1_score, precision, recall, rmse, mse
* Normalization class with mean normalization, centering, min-max normalization
* Batch Gradient descent optimization algorithms


```To-do list```
* Linear Support Vector Machines
* Kernel Support Vector Machine
* Anomaly Detection Algorithm
* Recommender Systems Algorithm
* Mini-batch Gradient Descent


## Importing Algorithms

on your terminal, change to folder where you downloaded the files
e.g.  

```
cd path_to_folder/  
from learning_algorithms import linear_regression   
from learning_algorithms import scorer
```

```
lr = linear_regression.linearReg()  
lr.fit(X, y)  
pred = lr.predict(Xval)
```   

where (X,y) are the training examples and (Xval,yval) are the validation examples
```
sc = scorer.Scorer()  
sc.rmse(yval, pred)
```  

### Vectorization

Vectorized implementation of algorithms where used whenever possible to make codes run faster. Batch Gradient Descent used in all algorithms. Working on a mini-batch version.

## Author

* **Sewade Ogun** - [learning_algorithms](https://github.com/ogunlao/learning_algorithms)

## Acknowledgments

* Inspired to write from scratch after going through the <a href="https://www.coursera.org/learn/machine-learning">Machine Learning Course by Stanford Universiy</a> on Coursera, taught by Andrew Ng
* Special thanks to <a href="https://github.com/dibgerge/ml-coursera-python-assignments">dibgerge</a> for rewriting the course assignments in python. It encouraged me to use my already familiar language of python to understand machine learning concepts.