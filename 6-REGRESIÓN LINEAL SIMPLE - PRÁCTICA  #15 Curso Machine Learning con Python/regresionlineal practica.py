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
from sklearn import datasets, linear_model
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
    
############### PREPARAR LA DATA REGRESION LINEAL SIMPLE ###############

#Seleccionamos solamente la columna 5 del dataset (RM que es el numero de habitaciones que tiene la casa)
x = boston.data[:, np.newaxis, 5]

#Definimos los datos correspondientes a las etiquetas
y = boston.target

#Graficamos los datos correspondientes
plt.scatter(x,y)
plt.xlabel('Numero de habitaciones')
plt.ylabel('Valor medio')
plt.show()

############### IMPLEMENTACION DE REGRESION LINEAL SIMPLE ###############

from sklearn.model_selection import train_test_split

#Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

#Definimos el algoritmo a utilizar
lr = linear_model.LinearRegression()

#Entrenamos el modelo
lr.fit(x_train, y_train)

#Realizo una prediccion
y_pred = lr.predict(x_test)

#Graficamos los datos junto con el modelo
plt.scatter(x_test, y_test)
plt.plot(x_test, y_pred, color='red', linewidth=3)
plt.title('Regresion Lineal Simple')
plt.xlabel('Numero de habiraciones')
plt.ylabel('Valor medio')
plt.show()

print()
print('DATOS DEL MODELO DE REGRESION LINEAL SIMPLE')
print()

print('Valor de la pendiente o coeficiente "a":')
print(lr.coef_)
print()

print('Valor de la interseccion u ordenada "b":')
print(lr.intercept_)
print()

print('La ecuacion del modelo es iugal a:')
print('y = ', lr.coef_, 'x = ', lr.intercept_)
print()

print('Precision del modelo:')
print(lr.score(x_train, y_train))


































