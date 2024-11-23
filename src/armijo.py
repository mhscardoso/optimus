import numpy as np
from typing import Callable

def armijo(
    f:  Callable[[np.ndarray], float],
    df: Callable[[np.ndarray], np.ndarray],
    x:  np.ndarray, 
    d:  np.ndarray, 
    gamma: float, 
    ni: float,
):
    t = 1
    dx = df(x)

    while f(x + t * d) > f(x) + ni * t * np.dot(dx, d.T):
        t *= gamma
    
    return t
