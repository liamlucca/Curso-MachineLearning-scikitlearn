 # -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sklearn import linear_model



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
    
############### PREPARAR LA DATA REGRESION LINEAL MULTIPLE ###############

#Seleccionamos las columna 5, 6, 7 del dataset (RM = numero de habitaciones, AGE = Antiguedad, DIS = distancia que se encuentra de los centros de trabajo de boston)
x_multiple = boston.data[:, 5:8]

#Definimos los datos correspondientes a las etiquetas
y_multiple = boston.target

############### IMPLEMENTACION DE REGRESION LINEAL SIMPLE ###############

from sklearn.model_selection import train_test_split

#Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
x_train, x_test, y_train, y_test = train_test_split(x_multiple, y_multiple, test_size=0.2)

#Definimos el algoritmo a utilizar
lr_multiple = linear_model.LinearRegression()

#Entrenamos el modelo
lr_multiple.fit(x_train, y_train)

#Realizo una prediccion
y_pred_multiple = lr_multiple.predict(x_test)

print()
print('DATOS DEL MODELO DE REGRESION LINEAL MULTIPLE')
print()

print('Valor de las pendientes o coeficientes "a":')
print(lr_multiple.coef_)
print()

print('Valor de la interseccion u ordenada "b":')
print(lr_multiple.intercept_)
print()

print('La ecuacion del modelo es iugal a:')
print('y = ', lr_multiple.coef_, 'x = ', lr_multiple.intercept_)
print()

print('Precision del modelo:')
print(lr_multiple.score(x_train, y_train))


































