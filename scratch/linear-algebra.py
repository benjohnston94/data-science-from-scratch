"""
Directory:
- Vector
- Add two vectors
- Subtract two vectors
- Vector sum (component-wise of multiple vectors)
- vector mean
- Scalar multiply
- Dot
- sum of sqaures
- magnitude
- distance
"""

"""
A Vector is just a one dimensional array
"""
from numpy import array as Vector

def add(v: Vector, w: Vector) -> Vector:
    """
    Add the components of two vectors 
    The i dimensions will add together
    Vectors must be the same length
    """
    assert len(v) == len(w), "Vectors must be the same length!"

    return [v[i] + w[i] for i in range(len(v))]

assert add([1, 2, 3], [2, 4, 6]) == [3, 6, 9]

