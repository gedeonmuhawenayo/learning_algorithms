# Project Title

Learning Algorithms written from scratch using numpy.

## Getting Started

Clone the repo 
    git clone https://github.com/ogunlao/learning_algorithms.git

or download a zipped version to use on your local machine.

### Prerequisites

* python >= 3.6
* numpy >= 1.14

```
Tutorials and how to download Python can be found on <a href="https://www.python.org/">Python website<a> while numpy can be gotten 
from <a href="https://www.numpy.org/">numpy website</a> 
```

### Contents

* Linear Regression algorithm with regularization
* Logistic Regression
* Scoring class which includes Accuracy, f1_score, precision, recall, rmse, mse
* Normalization class with mean normalization, centering, min-max normalization

```To-do list```
* Linear Support Vector Machines
* Kernel Support Vector Machines
* Neural Networks Algorithm
* Back Propagation
* Principal Component Analysis
* k-means Algorithm
* Anomaly Detection Algorithm
* Recommender Systems Algorithm


## Importing Algorithms

on your terminal, change to folder where you downloaded the files
e.g.
cd path_to_file/
from learning_algorithms import linear_regression
from learning_algorithms import logistic_regression
from learning_algorithms import scorer
e.t.c

lr = linear_regression.linearReg()
lr.fit(X, y)
pred = lr.predict(Xval)

where (X,y) are the training examples and (Xval,yval) are the validation examples

sc = scorer.Scorer()
sc.rmse(yval, pred)

### Vectorization

In all cases where I was aware of vectorized version of arithmetics, I avoided loops except where necessary.

## Author

* **Sewade Ogun** - [learning_algorithms](https://github.com/ogunlao)

## Acknowledgments

* Inspired to write from scratch after going through the <a href="https://www.coursera.org/learn/machine-learning">Machine Learning Course by Stanford Universiy</a> on Coursera, taught by Andrew Ng
* Special thanks to <a herf="https://github.com/dibgerge/ml-coursera-python-assignments">dibgerge</a> for rewriting the course assignments in python. It encouraged me to use my already familiar language of python to understand machine learning concepts.

