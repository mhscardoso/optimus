import numpy as np

from src.functions.f1 import DIMENSION as d1, f1, Df1
from src.functions.f2 import DIMENSION as d2, f2, Df2
from src.functions.f3 import DIMENSION as d3, f3, Df3

from src.gradient.gradient import gradient

d  = d2
f  = f2
df = Df2

GAMMA = 0.9
NI = 0.4

mult_factor = np.zeros(d)
for i in range(d):
    mult_factor[i] = np.random.randint(100)

x1_f1 = np.random.rand(d) * mult_factor

# x1_f1 = np.array([0.07304271, 0.75131362, 3.54007304, 0.15500378, 8.61282159, 0, 4.16660364])
# x1_f1 = np.array([1.92883955, 3.46330774, 3.93182327, 2.66653325, 0.13257894, 1.52447945, 1.07824138])
# x1_f1 = np.array([0.30413396, 0.77326552, 0.4208701, 0.76678017, 0.31038108, 0.23579468, 0.68483634])

print(f'Ponto Inicial: {x1_f1}')

min_x, stop, k = gradient(x1_f1, f, df, GAMMA, NI)

if stop == 1:
    print(f'Critério de Parada: Gradiente\n')
elif stop == 2:
    print(f'Critério de Parada: xk+1 = xk\n')
elif stop == 3:
    print('Critério de Parada: Número de Iterações')


print(f'Iterações: {k}')
print(f'x_ = {min_x}')
print(f'Ótimo: {f(min_x)}')
