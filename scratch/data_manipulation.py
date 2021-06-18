"""
Various functions for working with data
"""
from typing import List, Tuple
from linear_algebra import Vector, vector_mean
from statistics import standard_deviation

def scale(data: List[Vector]) -> Tuple[Vector, Vector]:
    "Return the column-wise mean and standard deviation of a dataset"
    dim = len(data[0])

    means = vector_mean(data)
    std_devs = [standard_deviation([vector[i] for vector in data])
                for i in range(dim)]

    return means, std_devs

test_data = [
    [1, 1, 3],
    [-1, 3, 1],
    [0, 5, 2]
]
means, std_devs = scale(test_data)
assert means == [0, 3, 2]
assert std_devs == [1, 2, 1]


def rescale(data: List[Vector]) -> List[Vector]:
    """
    Rescale data so it has a mean of 0 and standard deviation of 1
    To do this we subtract the mean and divide by the standard deviation
    """
    dim = len(data[0])
    means, std_devs = scale(data)

    # copy the data so we don't overwrite it
    rescaled_data = [vector[:] for vector in data]

    for v in rescaled_data:
        for i in range(dim):
            if std_devs[i] > 0:
                v[i] = (v[i] - means[i]) / std_devs[i]
    
    return rescaled_data

# check the mean and std deviations of the test data look as expected
means, std_devs = scale(rescale(test_data))
assert means == [0,0,0]
assert std_devs == [1,1,1]
