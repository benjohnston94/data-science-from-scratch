"""
Steps for logistic regression and the functions required

LINEAR FUNCTION
- Initialise random betas for each feature
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
from scratch.linear-algebra import Vector

def logistic(x: float) -> float:
    """
    Take a number and returns a value between 0 and 1
    """
    return 1 / (1 + math.exp(x))

for x in range(-30, 30):
    assert 0 < logistic(X) < 1, "logistic should return a value between 0 and 1"
