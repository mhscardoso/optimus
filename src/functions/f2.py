import numpy as np
import matplotlib.pyplot as plt 

DIMENSION     = 100
vec_function  = np.vectorize(lambda x: x**4 - 16 * x**2 + 5 * x)
dvec_function = np.vectorize(lambda x: 4 * x**3 - 32 * x + 5)


def plot_aux(L: float):
    x = np.linspace(-L, L)
    y = vec_function(x)
    dy = dvec_function(x)

    plt.title("Função aux $f_2(x)$")
    plt.xlim(-20, 20)
    plt.ylim(-90, 20)
    plt.xlabel("X")
    plt.grid(visible=True)

    plt.plot(x, y, color = 'red')
    plt.plot(x, dy, color = 'blue')
    plt.show()


def f2(x: np.ndarray) -> float:
    ps = vec_function(x)

    return np.sum(ps)


def Df2(x: np.ndarray) -> np.ndarray:
    result = dvec_function(x)
    
    return result


def Hf2(x: np.ndarray) -> np.ndarray:
    result = np.zeros((DIMENSION, DIMENSION))

    for i in range(DIMENSION):
        result[i][i] = 12 * x[i]**2 - 32
    
    return result

