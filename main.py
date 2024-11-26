import time
import numpy as np

from src.functions.f1 import F1
from src.functions.f2 import F2
from src.functions.f3 import F3

from src.gradient.gradient import gradient
from src.newton.newton import newton
from src.anewton.anewton import anewton_dfp

f = F3()

GAMMA = 0.9
NI = 0.4

mult_factor = np.random.randint(-5, 5, f.DIMENSION)
x1_f1 = np.random.rand(f.DIMENSION) * mult_factor

start_time = time.time()
min_x, stop, k, k_armijo = anewton_dfp(x1_f1, f, NI, GAMMA)
end_time = time.time()

exec_time = end_time - start_time
f_min_x = f.f(min_x)

print('------------------------------------------------')
print(f'Tempo de Execução = {exec_time} s')
print(f'# Iterações       = {k}')
print(f'# Armijo          = {k_armijo}\n')
print(f'x_                = {min_x}')
print(f'Ótimo             = {f_min_x}\n')
print(f'Erro              = {np.linalg.norm(f.df(min_x))}')
print('Ponto Inicial     =')

print('[', end='')
if f.DIMENSION == 7:
    print(f' {x1_f1[0]}, ')
    for x in x1_f1:
        print(f'  {x}, ')
elif f.DIMENSION == 100:
    print(x1_f1)
elif f.DIMENSION == 2:
    print(f'{x1_f1[0]}, {x1_f1[1]}')
print(']')


if stop == 1:
    print(f'Critério de Parada: Gradiente')
elif stop == 2:
    print(f'Critério de Parada: xk+1 = xk')
elif stop == 3:
    print('Critério de Parada: Número de Iterações')
elif stop == 4:
    print('Critério de Parada: Divisão por zero')

print('--------------------- END ----------------------\n')
