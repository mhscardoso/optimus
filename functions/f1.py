import numpy as np
import numpy.typing as npt

DIMENSION_LIMIT = 6

def f1(x: npt.ArrayLike):
    result = 0

    # From 0 to 5
    for i in range(DIMENSION_LIMIT - 1):
        result += (100 * (x[i+1] - x[i] ** 2)) ** 2 + (1 - x[i]) ** 2
    
    return result




