import numpy as np
from src.functions.base import Function
from typing import Callable

def armijo(
    f:     Function,
    x:     np.ndarray, 
    d:     np.ndarray, 
    ni:    float,
    gamma: float, 
):
    t   = 1
    fx  = f.f(x)
    dx  = f.df(x)
    dot = dx @ d
    k   = 0

    while f.f(x + t * d) > (fx + (ni * t * dot)):
        t *= gamma
        k += 1
    
    return t, k
