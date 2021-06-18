"""
Directory:
- Vector
- Add two vectors
- Subtract two vectors
- Vector sum (component-wise of multiple vectors)
- Scalar multiply
- vector mean
- Dot
- sum of sqaures
- magnitude
- distance
"""
from typing import List

# these should later be turned into classes
Vector = List[float]  # A Vector is a one dimensional array
Matrix = List[Vector]  # A Matrix is list of vectors


def add(v: Vector, w: Vector) -> Vector:
    """
    Add the components of two vectors 
    Vectors must be the same length
    """
    assert len(v) == len(w), "Vectors must be the same length!"

    return [v[i] + w[i] for i in range(len(v))]


assert add([1, 2, 3], [2, 4, 6]) == [3, 6, 9]
assert not add([1, 2, 3], [1, 2, 3]) == [1, 1, 1]


def subtract(v: Vector, w: Vector) -> Vector:
    """
    Subtract the components of two vectors
    vectors must be the same length
    """
    assert len(v) == len(w), "Vectors must be the same length!"

    return [v[i] - w[i] for i in range(len(v))]


assert subtract([1, 2, 3], [2, 4, 6]) == [-1, -2, -3]


def vector_sum(matrix: Matrix) -> Vector:
    """
    Takes in a matrix (2d array of vectors)
    Adds up elements of all vectors component-wise
    All vectors must be the same length
    """
    assert matrix, "Matrix must not be empty!"
    n_dims = len(matrix[0])
    assert all(len(v) == n_dims for v in matrix), "all vectors must be the same length"

    return [sum(v) for v in zip(*matrix)]


assert vector_sum([[1, 1, 1],
                   [1, 2, 3],
                   [2, 2, 2]]) == [4, 5, 6]


def scalar_multiply(v: Vector, x: float) -> Vector:
    """
    Multiplies each component of a vector by
    a constant x
    """
    return [v[i] * x for i in range(len(v))]


assert scalar_multiply([1, 2, 3], 2) == [2, 4, 6]
assert not scalar_multiply([1, 2, 3], 2) == [1, 2, 3]


def vector_mean(matrix: List[Vector]) -> Vector:
    """
    Takes in a list of n-dimensional vectors
    Returns a vector with the component-wise mean of input vectors
    """
    assert matrix, "Matrix must not be empty!"
    m_len = len(matrix)
    # assert all(len(v) == n_dims for v in matrix), "all vectors must be the same length"

    v_summed = vector_sum(matrix)

    return scalar_multiply(v_summed, 1 / m_len)


assert vector_mean([[1, 2, 3],
                    [2, 3, 4],
                    [3, 4, 8]]) == [2, 3, 5]


def dot(v: Vector, w: Vector) -> float:
    """
    Multiply two vector component-wise and
    returns the sum of the results
    """
    assert len(v) == len(w), "vectors must be the same length!"

    return sum(v_i * w_i for v_i, w_i in zip(v, w))


assert dot([2, 3, 4], [2, 2, 2]) == 18


def sum_of_squares(v: Vector) -> float:
    """
    Returns the sum of squares for a vector
    i.e. the a vector dotted with itself
    """
    return dot(v, v)


assert sum_of_squares([2, 2, 2]) == 12

    
def gradient_step(weights: Vector,
                  gradients: Vector,
                  step_size: float):
    """
    Takes in current weights, the gradient of the loss functions, and a step size
    Multiplies the gradients by the step size and adds this to the weights to produce
    a set of updated values (i.e. takes a 'step' in that direction)
    NOTE: need to think about the best place to put the negative (currently multiplying by step_size)
    """
    step = scalar_multiply(gradients, step_size)
    return add(weights, step)


assert gradient_step([10, 20, 30], [1, 2, 3], -0.1) == [9.9, 19.8, 29.7]
