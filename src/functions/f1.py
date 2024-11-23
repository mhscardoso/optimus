import numpy as np

DIMENSION = 7

def f1(x: np.ndarray) -> float:
    result = 0

    # From 0 to 5
    for i in range(DIMENSION - 2):
        result += 100 * ((x[i+1] - x[i] ** 2) ** 2) + ((1 - x[i]) ** 2)
    
    return result


def Df1(x: np.ndarray) -> np.ndarray:
    result = np.zeros(DIMENSION)
    result[0] = 400 * x[0]**3 - 400 * x[0] * x[1] + 2 * x[0] - 2

    for i in range(1, DIMENSION - 1):
        result[i] = 400 * x[i]**3 - 400 * x[i] * x[i+1] - 200 * x[i-1]**2 + 202 * x[i] - 2
    
    result[6] = -200 * x[5]**2 + 200 * x[6]

    return result


def Hf1(x: np.ndarray) -> np.ndarray:
    result = np.zeros((DIMENSION, DIMENSION))

    result[0][0] = 1200 * x[0]**2 - 400 * x[1] + 2
    result[0][1] = -400 * x[0]

    result[6][5] = -400 * x[5]
    result[6][6] = 200

    for i in range(1, DIMENSION - 1):
        result[i][i-1] = -400 * x[i-1]
        result[i][i]   = 1200 * x[i]**2 - 400 * x[i+1] + 202
        result[i][i+1] = -400 * x[i]
    
    return result
