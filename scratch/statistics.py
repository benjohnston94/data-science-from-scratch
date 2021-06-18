"""
Various statistical functions. 

To build:
- Mean
- Variance
- Standard deviation
- PDF
- CDF
"""
from typing import List
from linear_algebra import sum_of_squares
import math

def mean(xs: List[float]) -> float:
    "Mean of 1d vector"
    return sum(xs) / len(xs)

def de_mean(xs: List[float]) -> List[float]:
    "Centre data around mean of 0"
    x_bar = mean(xs)
    return [x - x_bar for x in xs]

def variance(xs: List[float]) -> float:
    "Average squared difference from the mean (however we divide by n-1 instead of n)"
    n = len(xs)

    deviations = de_mean(xs)
    return sum_of_squares(deviations) / (n-1)

def standard_deviation(xs: List[float]) -> float:
    return math.sqrt(variance(xs))

    
