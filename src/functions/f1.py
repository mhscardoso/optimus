import numpy as np
from src.functions.base import Function

class F1(Function):

    DIMENSION = 7

    def f(self, x: np.ndarray) -> float:
        result = 0

        for i in range(self.DIMENSION - 2):
            result += 100 * ((x[i+1] - x[i] ** 2) ** 2) + ((1 - x[i]) ** 2)
        
        return result


    def df(self, x: np.ndarray) -> np.ndarray:
        result = np.zeros(self.DIMENSION)
        result[0] = (400 * x[0]**3) - (400 * x[0] * x[1]) + (2 * x[0]) - 2

        for i in range(1, self.DIMENSION - 1):
            result[i] = (400 * x[i]**3) - (400 * x[i] * x[i+1]) - (200 * x[i-1]**2) + (202 * x[i]) - 2
        
        result[6] = (-200 * x[5]**2) + (200 * x[6])

        return result


    def hf(self, x: np.ndarray) -> np.ndarray:
        result = np.zeros((self.DIMENSION, self.DIMENSION))

        result[0][0] = 1200 * x[0]**2 - 400 * x[1] + 2
        result[0][1] = -400 * x[0]

        result[6][5] = -400 * x[5]
        result[6][6] = 200

        for i in range(1, self.DIMENSION - 1):
            result[i][i-1] = -400 * x[i-1]
            result[i][i]   = 1200 * x[i]**2 - 400 * x[i+1] + 202
            result[i][i+1] = -400 * x[i]
        
        return result
