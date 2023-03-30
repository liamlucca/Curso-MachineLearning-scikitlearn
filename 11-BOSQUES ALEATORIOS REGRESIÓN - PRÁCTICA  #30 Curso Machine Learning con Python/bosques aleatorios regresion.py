# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sklearn import linear_model

# Regresion Lineal Simple
# y = ax + b


############### LIBRERIAS ###############

# Se importan las librerias a utilizar
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

############### PREPARAR LA DATA ###############

#Importamos los datos de la misma libreria de scikit-learn
boston = datasets.load_boston()
print(boston)
print()

############### ENTENDIMIENTO DE LA DATA ###############

#Verifico la informacion contenida en el dataset
print('Informacion en el dataset:')
print(boston.keys())
print()

#Verifico las caracteristicas del dataset
print('Caracteristicas del dataset:')
print(boston.DESCR)
print()

#Verifico la cantidad de datos que hay en los dataset
print('Cantidad de datos:')
print(boston.data.shape)
print()

#Verifico la informacion de las columnas
print(boston.feature_names)
print()
    
############### PREPARAR LA DATA BOSQUES ALEATORIOS REGRESION ###############

#Seleccionamos solamente la columna 5 del dataset (RM que es el numero de habitaciones que tiene la casa)
x = boston.data[:, np.newaxis, 5]

#Definimos los datos correspondientes a las etiquetas
y = boston.target

#Graficamos los datos correspondientes
plt.scatter(x,y)
plt.xlabel('Numero de habitaciones')
plt.ylabel('Valor medio')
plt.show()

############### IMPLEMENTACION DE BOSQUES ALEATORIOS REGRESION ###############

from sklearn.model_selection import train_test_split

#Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

from sklearn.ensemble import RandomForestRegressor

#Definimos el algoritmo a utilizar
bar = RandomForestRegressor(n_estimators= 300, max_depth= 8)

#Entrenamos el modelo
bar.fit(x_train, y_train)

#Realizo una prediccion
y_pred = bar.predict(x_test)

#Graficamos los datos de prueba junto con la prediccion
x_grid = np.arange(min(x_test), max(x_test), 0.1)
x_grid = x_grid.reshape(len(x_grid),1)
plt.scatter(x_test, y_test)
plt.plot(x_grid, bar.predict(x_grid), color='red', linewidth=3)
plt.show()

print()
print('DATOS DEL MODELO DE BOSQUES ALEATORIOS REGRESION')
print()

print('Precision del modelo:')
print(bar.score(x_train, y_train))


































