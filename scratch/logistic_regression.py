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
from linear_algebra import Vector, Matrix, dot

def logistic(x: float) -> float:
    """
    Take a number and returns a value between 0 and 1
    """
    return 1 / (1 + math.exp(-x))

# a negative input transform by logistic should return a value between 0 and 0.5
# a positive input should return a value between 0.5 and 1
assert 0 < logistic(-30) < 0.5
assert 0 < logistic(0) == 0.5
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
