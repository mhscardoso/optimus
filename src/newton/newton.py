import numpy as np
from numpy.linalg import inv
from src.functions.base import Function
from src.common import MIN_FLOAT
from src.armijo import armijo

def newton(
    x0:    np.ndarray,
    f:     Function,
    ni:    float,
    gamma: float,
):
    k = 0
    k_armijo = 0
    xk = x0
    stop = 0

    while True:
        dkm = f.df(xk)
        hfx = inv(f.hf(xk))

        # --- Método de Newton --- #
        dk = - hfx @ dkm           #
        # ------------------------ #

        tk, k_a = armijo(f, xk, dk, ni, gamma)
        xkm1 = xk + tk * dk
        diff = xkm1 - xk
        xk = xkm1
        k_armijo += k_a
        k += 1

        if np.all(np.absolute(dkm) < MIN_FLOAT):
            stop = 1
            break

        if np.all(np.absolute(diff) < MIN_FLOAT):
            stop = 2
            break

        if k > 40000:
            stop = 3
            break
    
    return xk, stop, k, k_armijo
