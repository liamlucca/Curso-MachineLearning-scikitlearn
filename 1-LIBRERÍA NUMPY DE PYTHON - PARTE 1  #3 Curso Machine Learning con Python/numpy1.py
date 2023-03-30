import numpy as np
import sys
import time

# Prueba de numpy

a1= np.array([1,2,3])
print('1D array:')
print(a1)
print()

a2= np.array([(1,2,3),(4,5,6)])
print('2D array:')
print(a2)
print()

# Comparacion de tamaño entre listas de python y arrays de numpy

b1 = range(1000)
print('Tamaño de lista de Python:')
print(sys.getsizeof(5)*len(b1))
print()

b2 = np.arange(1000)
print('Tamaño de NumPy array:')
print(b2.size*b2.itemsize)
print()

# Comparacion de tiempo entre listas de python y arrays de numpy

size_c = 1000000

c1 = range(size_c)
c2 = range(size_c)

c3 = np.arange(size_c)
c4 = np.arange(size_c)

start = time.time()
result = [(x,y) for x,y in zip(c1,c2)]
print('Tiempo Python')
print((time.time()-start)*1000)
print()

start = time.time()
result = c3+c4
print('Tiempo Numpy')
print((time.time()-start)*1000)


