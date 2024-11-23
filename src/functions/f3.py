import numpy as np
import matplotlib.pyplot as plt

DIMENSION = 2

aux_vec_function = np.vectorize(lambda x: x**3 + 3*x**2 - 13*x - 38)

def plot_aux(L: float):
    x = np.linspace(-L, L)
    y = aux_vec_function(x)

    plt.title("$x^3 + 3x^2 - 13x - 38$")
    plt.xlim(-7, 5)
    plt.ylim(-50, 30)
    plt.xlabel("X")
    plt.grid(visible=True)

    plt.plot(x, y, color = 'red')
    plt.show()


def f3(x: np.ndarray) -> float:
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2


def Df3(x: np.ndarray) -> np.ndarray:
    result = np.zeros(2)
    result[0] = 4 * x[0]**3 + 4 * x[0] * x[1] - 42 * x[0] + 2 * x[1]**2 - 14
    result[1] = 4 * x[1]**3 + 4 * x[0] * x[1] - 26 * x[1] + 2 * x[0]**2 - 22

    return result


def Hf3(x: np.ndarray) -> np.ndarray:
    result = np.zeros((DIMENSION, DIMENSION))

    result[0][0] = 12 * x[0]**2 + 4 * x[1] - 42
    result[0][1] = 4 * (x[0] + x[1])
    result[1][0] = 4 * (x[0] + x[1])
    result[1][1] = 12 * x[1]**2 + 4 * x[0] - 26

    return result


