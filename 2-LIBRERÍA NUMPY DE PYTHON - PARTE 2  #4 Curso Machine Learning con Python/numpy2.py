# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 14:10:31 2021

@author: liaml
"""

import numpy as np

# Crear una matriz de unos - 3 filas 4 columnas

print('Crear una matriz de unos - 3 filas 4 columnas')

unos = np.ones((3,4))
print(unos)
print()

# Crear una matriz de ceros - 3 filas 4 columnas

print('Crear una matriz de ceros - 3 filas 4 columnas')

ceros = np.zeros((3,4))
print(ceros)
print()

# Crear una matriz con valores aleatorios

print('Crear una matriz con valores aleatorios')

aleatorios =np.random.random((2,2))
print(aleatorios)
print()

# Crear una matriz vacia

print('Crear una matriz vacia')

vacia = np.empty((3,2))
print(vacia)
print()

# Crear una matriz con un solo valor

print('Crear una matriz con un solo valor')

full = np.full((2,2),8)
print(full)
print()

# Crear una matriz con valores espaciados uniformemente

print('Crear una matriz con valores espaciados uniformemente')

espacio1 = np.arange(0,30,5) # empieza del 0 hasta el 30 de 5 en 5
print(espacio1)

espacio2 = np.linspace(0,2,5) # empieza del 0, llegando al 2 de 0,5 en 0,5 (creo)
print(espacio2)

print()

# Crear una matriz identidad: una matriz cuadrada en la cual todos los elementos en la diagonal principal son 1

print('Crear una matriz identidad: una matriz cuadrada en la cual todos los elementos en la diagonal principal son 1')

identidad1 = np.eye(4,4)
print(identidad1)

identidad2 = np.eye(4)
print(identidad2)

print()

# Conocer las dimensiones de una matriz

print('Conocer las dimensiones de una matriz')

a = np.array([(1,2,3), (4,5,6)])
print(a.ndim)
print()

# Conocer el tipo de los datos

print('Conocer el tipo de los datos')

b = np.array([(1,2,3)])
print(b.dtype)
print()

# Conocer el tamaño y forma de la matriz

print('Conocer el tamaño y forma de la matriz')

c = np.array([(1,2,3,4,5,6)])
print(c.size)
print(c.shape)
print()

# Cambio de forma de una matriz

print('Cambio de forma de una matriz')

d = np.array([(8,9,10), (11,12,13)])
print(d)

d = d.reshape(3,2)
print(d)

print()

# Extraer un solo valor de la matriz - el valor ubicado en la fila 0 columna 2

print('Extraer un solo valor de la matriz - el valor ubicado en la fila 0 columna 2')

e = np.array([(1,2,3,4), 
              (3,4,5,6)])
print(e[0,2])
print()

# Extraer los valores de todas las filas ubicados en la columna 3

print('Extraer los valores de todas las filas ubicados en la columna 3')

print(e[0:,2])
print()

# Encontrar el minimo, maximo y la suma

print('Encontrar el minimo, maximo y la suma')

f= np.array([2,4,8])
print(f.min())
print(f.max())
print(f.sum())
print()

# Calcular la raiz cuadrada y la desviacion estandar

print('Calcular la raiz cuadrada y la desviacion estandar')

g = np.array([(1,2,3), (3,4,5)])
print(np.sqrt(g))
print(np.std(g))
print()

# Calcular la suma, resta, multiplicacion y division de dos matrices

print('Calcular la suma, resta, multiplicacion y division de dos matrices')

x = np.array([(1,2,3),(3,4,5)])
y = np.array([(1,2,3),(3,4,5)])

print(x+y)
print(x-y)
print(x*y)
print(x/y)






