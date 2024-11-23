# import sys
import numpy as np
from typing import Callable
from src.armijo import armijo

MIN_FLOAT = 10e-13

def gradient(
    x0: np.ndarray,
    f:  Callable[[np.ndarray], float],
    df: Callable[[np.ndarray], np.ndarray],
    gamma: float,
    ni: float,
):
    k = 0
    xk = x0
    dkm = df(xk)
    stop = 0

    while True:
        dkm = df(xk)

        # --- Método do Gradiente --- #
        dk = (-1) * dkm               #
        # --------------------------- #

        tk = armijo(f, df, xk, dk, gamma, ni)
        xkm1 = xk + tk * dk
        diff = xkm1 - xk
        xk = xkm1
        k += 1

        if len(dkm[np.abs(dkm) > MIN_FLOAT]) == 0:
            stop = 1
            break

        if len(diff[np.abs(diff) > MIN_FLOAT]) == 0 and k > 10000:
            stop = 2
            break

        if k > 10000:
            stop = 3
            break
    
    return xk, stop, k
