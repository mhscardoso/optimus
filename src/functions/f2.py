import numpy as np
import matplotlib.pyplot as plt 
from src.functions.base import Function

class F2(Function):
    DIMENSION     = 100
    vec_function  = np.vectorize(lambda x: x**4 - 16 * x**2 + 5 * x)
    dvec_function = np.vectorize(lambda x: 4 * x**3 - 32 * x + 5)

    def f(self, x: np.ndarray) -> float:
        ps = self.vec_function(x)

        return np.sum(ps)


    def df(self, x: np.ndarray) -> np.ndarray:
        result = self.dvec_function(x)
        
        return result


    def hf(self, x: np.ndarray) -> np.ndarray:
        result = np.zeros((self.DIMENSION, self.DIMENSION))

        for i in range(self.DIMENSION):
            result[i][i] = 12 * x[i]**2 - 32
        
        return result


    def plot_aux(self, L: float):
        x = np.linspace(-L, L)
        y = self.vec_function(x)
        dy = self.dvec_function(x)

        plt.title("Função aux $f_2(x)$")
        plt.xlim(-20, 20)
        plt.ylim(-90, 20)
        plt.xlabel("X")
        plt.grid(visible=True)

        plt.plot(x, y, color = 'red')
        plt.plot(x, dy, color = 'blue')
        plt.show()
