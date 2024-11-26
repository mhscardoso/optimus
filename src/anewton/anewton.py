import numpy as np
from numpy.linalg import inv
from src.functions.base import Function
from src.common import MIN_FLOAT
from src.armijo import armijo

def multi(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    dimension = len(p)
    result = np.zeros((dimension, dimension))

    for lines in range(dimension):
        for columns in range(dimension):
            result[lines][columns] = p[lines] * q[columns]
    
    return result


def dfp(p: np.ndarray, q: np.ndarray, h0: np.ndarray) -> np.ndarray:
    am1 = ((p.T).dot(q))
    am2 = (np.dot(np.dot(q, h0), q))

    if am1 == 0 or am2 == 0:
        return None

    m1 = 1 / am1
    m2 = 1 / am2

    t1 = (multi(p, p)) * m1
    t2 = np.dot(multi(np.dot(h0, q), q), h0) * m2

    return h0 + t1 - t2


def anewton_dfp(
    x0:    np.ndarray,
    f:     Function,
    ni:    float,
    gamma: float,
):
    k = 0
    k_armijo = 0
    xk = x0
    stop = 0
    hfx = inv(f.hf(xk))

    while True:
        dkm = f.df(xk)

        # --- Método de Quase-Newton --- #
        dk = - hfx @ dkm                 #
        # ------------------------------ #

        tk, k_a = armijo(f, xk, dk, ni, gamma)
        k_armijo += k_a

        xkm1 = xk + tk * dk
        pk = xkm1 - xk
        qk = f.df(xkm1) - dkm

        hfx = dfp(pk, qk, hfx)

        xk = xkm1

        k += 1

        if np.all(np.absolute(dkm) < MIN_FLOAT):
            stop = 1
            break

        if np.all(np.absolute(pk) < MIN_FLOAT):
            stop = 2
            break

        if k > 40000:
            stop = 3
            break

        if hfx is None:
            stop = 4
            break

    
    return xk, stop, k, k_armijo
