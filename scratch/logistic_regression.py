"""
Steps for logistic regression and the functions required

LINEAR FUNCTION
- Initialise random weights for each feature
- Output of linear function equals 'log odds' of output
- Alternatively, the linear function transformed by 
the logistic function gives the output of the probability

LOSS FUNCTION (maximum likelihood)
- Error is as follows:
    If target = 1, 1 - p
    If target = 0, p
- Sum the negative log of these probabilities
    Negative because we'll minimise the loss with gradient descent
    Summing the logs of probabilities is better than multiplying lots of small probabilities
- Get the gradient of the loss function and take a step in this direction

FUNCTIONS REQUIRED (PYTHON)
- logistic function
- negative log likelihood (for individual and total)
- derivative of likelihood function
- gradient step (in a separate .py file)
"""

import math 
from linear_algebra import Vector, Matrix, dot, vector_sum, gradient_step
from typing import List
import random

def logistic(x: float) -> float:
    """
    Take a number and returns a value between 0 and 1
    """
    return 1 / (1 + math.exp(-x))

# a negative input transform by logistic should return a value between 0 and 0.5
# a positive input should return a value between 0.5 and 1
assert 0 < logistic(-30) < 0.5
assert logistic(0) == 0.5
assert 0.5 < logistic(30) < 1


def _negative_log_likelihood(x: Vector,
                             y: float,
                             weights: Vector) -> float:
    """
    Return negative log likehood of an input (dotted by function weights)
    The dot product transform by the logistic function is 'p'
    Therefore for y = 1, we will return 1 - p
    and if y = 0, we will just return p
    """
    if y == 1:
        return -math.log(1 - logistic(dot(x, weights)))
    elif y == 0:
        return -math.log(logistic(dot(x, weights)))
    else:
        raise ValueError(f"invalid y value: {y}")


def negative_log_likelihood(X: Matrix,
                            y: Vector,
                            weights: Vector) -> float:
    return sum(_negative_log_likelihood(x, y, weights)
                for x, y in zip(X, y))


"""
Now to compute the gradient for this function
This requires three functions:
1. Get partial derivative for a given feature
2. Get the gradient for one data point
3. Get the gradient for all points (which just requires summing)
"""

def _negative_log_partial_derivative(x: Vector,
                                     y: float,
                                     beta: Vector,
                                     j: int) -> float:
    """Calculate jth partial derivative for one row"""
    return -(y - logistic(dot(x, beta))) * x[j]

def _negative_log_gradient(x: Vector,
                           y: float,
                           beta: Vector) -> Vector:
    "Gradient for one data point"
    return [_negative_log_partial_derivative(x, y, beta, j)
            for j in range(len(beta))]

def negative_log_gradient(xs: List[Vector],
                          ys: Vector,
                          beta: Vector) -> Vector:
    "Total 'error' i.e. summing the negative log gradients"
    return vector_sum([_negative_log_gradient(x, y, beta)
                       for x, y in zip(xs, ys)])


# This is all we need to put it all together
def logistic_regression(x_train: List[Vector],
                        y_train: Vector,
                        learning_rate: float = 0.001,
                        num_steps: int = 2000,
                        batch_size: int = 1) -> Vector:
    """
    Apply logistic regression using gradient descent
    to find the coefficients of the linear function
    """

    # Initialise random guess
    guess = [random.random() for _ in x_train[0]]
    
    # iterate 'num_steps' number of times
    for _ in range(num_steps):
        #iterate through data with steps of 'batch_size'
        for start in range(0, len(x_train), batch_size):
            # get number of rows according to batch_size
            batch_xs = x_train[start: start+batch_size]
            batch_ys = y_train[start: start+batch_size]

            # calculate 'mean' gradient across these points
            gradient = negative_log_gradient(batch_xs, batch_ys, guess)

            # update the 'guess' using gradient times the learning rate
            guess = gradient_step(guess, gradient, -learning_rate)

    return guess


# And that's it! Let's test it out now
if __name__=="__main__":

    # Copy and pasting sample data
    tuples = [(0.7,48000,1),(1.9,48000,0),(2.5,60000,1),(4.2,63000,0),(6,76000,0),(6.5,69000,0),(7.5,76000,0),(8.1,88000,0),(8.7,83000,1),(10,83000,1),(0.8,43000,0),(1.8,60000,0),(10,79000,1),(6.1,76000,0),(1.4,50000,0),(9.1,92000,0),(5.8,75000,0),(5.2,69000,0),(1,56000,0),(6,67000,0),(4.9,74000,0),(6.4,63000,1),(6.2,82000,0),(3.3,58000,0),(9.3,90000,1),(5.5,57000,1),(9.1,102000,0),(2.4,54000,0),(8.2,65000,1),(5.3,82000,0),(9.8,107000,0),(1.8,64000,0),(0.6,46000,1),(0.8,48000,0),(8.6,84000,1),(0.6,45000,0),(0.5,30000,1),(7.3,89000,0),(2.5,48000,1),(5.6,76000,0),(7.4,77000,0),(2.7,56000,0),(0.7,48000,0),(1.2,42000,0),(0.2,32000,1),(4.7,56000,1),(2.8,44000,1),(7.6,78000,0),(1.1,63000,0),(8,79000,1),(2.7,56000,0),(6,52000,1),(4.6,56000,0),(2.5,51000,0),(5.7,71000,0),(2.9,65000,0),(1.1,33000,1),(3,62000,0),(4,71000,0),(2.4,61000,0),(7.5,75000,0),(9.7,81000,1),(3.2,62000,0),(7.9,88000,0),(4.7,44000,1),(2.5,55000,0),(1.6,41000,0),(6.7,64000,1),(6.9,66000,1),(7.9,78000,1),(8.1,102000,0),(5.3,48000,1),(8.5,66000,1),(0.2,56000,0),(6,69000,0),(7.5,77000,0),(8,86000,0),(4.4,68000,0),(4.9,75000,0),(1.5,60000,0),(2.2,50000,0),(3.4,49000,1),(4.2,70000,0),(7.7,98000,0),(8.2,85000,0),(5.4,88000,0),(0.1,46000,0),(1.5,37000,0),(6.3,86000,0),(3.7,57000,0),(8.4,85000,0),(2,42000,0),(5.8,69000,1),(2.7,64000,0),(3.1,63000,0),(1.9,48000,0),(10,72000,1),(0.2,45000,0),(8.6,95000,0),(1.5,64000,0),(9.8,95000,0),(5.3,65000,0),(7.5,80000,0),(9.9,91000,0),(9.7,50000,1),(2.8,68000,0),(3.6,58000,0),(3.9,74000,0),(4.4,76000,0),(2.5,49000,0),(7.2,81000,0),(5.2,60000,1),(2.4,62000,0),(8.9,94000,0),(2.4,63000,0),(6.8,69000,1),(6.5,77000,0),(7,86000,0),(9.4,94000,0),(7.8,72000,1),(0.2,53000,0),(10,97000,0),(5.5,65000,0),(7.7,71000,1),(8.1,66000,1),(9.8,91000,0),(8,84000,0),(2.7,55000,0),(2.8,62000,0),(9.4,79000,0),(2.5,57000,0),(7.4,70000,1),(2.1,47000,0),(5.3,62000,1),(6.3,79000,0),(6.8,58000,1),(5.7,80000,0),(2.2,61000,0),(4.8,62000,0),(3.7,64000,0),(4.1,85000,0),(2.3,51000,0),(3.5,58000,0),(0.9,43000,0),(0.9,54000,0),(4.5,74000,0),(6.5,55000,1),(4.1,41000,1),(7.1,73000,0),(1.1,66000,0),(9.1,81000,1),(8,69000,1),(7.3,72000,1),(3.3,50000,0),(3.9,58000,0),(2.6,49000,0),(1.6,78000,0),(0.7,56000,0),(2.1,36000,1),(7.5,90000,0),(4.8,59000,1),(8.9,95000,0),(6.2,72000,0),(6.3,63000,0),(9.1,100000,0),(7.3,61000,1),(5.6,74000,0),(0.5,66000,0),(1.1,59000,0),(5.1,61000,0),(6.2,70000,0),(6.6,56000,1),(6.3,76000,0),(6.5,78000,0),(5.1,59000,0),(9.5,74000,1),(4.5,64000,0),(2,54000,0),(1,52000,0),(4,69000,0),(6.5,76000,0),(3,60000,0),(4.5,63000,0),(7.8,70000,0),(3.9,60000,1),(0.8,51000,0),(4.2,78000,0),(1.1,54000,0),(6.2,60000,0),(2.9,59000,0),(2.1,52000,0),(8.2,87000,0),(4.8,73000,0),(2.2,42000,1),(9.1,98000,0),(6.5,84000,0),(6.9,73000,0),(5.1,72000,0),(9.1,69000,1),(9.8,79000,1),]
    data = [list(row) for row in tuples]

    xs = [[1.0] + row[:2] for row in data]  # [intercept, variable_1, variable_2]
    ys = [row[2] for row in data]           # targer
    
    
