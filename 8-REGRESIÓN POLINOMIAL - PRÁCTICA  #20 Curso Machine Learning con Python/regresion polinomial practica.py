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
    
############### PREPARAR LA DATA REGRESION POLINOMIAL ###############

#Seleccionamos solamente la columna 6 del dataset
x_p = boston.data[:, np.newaxis, 5]

#Definimos los datos correspondientes a las etiquetas
y_p = boston.target

#Graficamos los datos correspondientes a las etiquetas
plt.scatter(x_p, y_p)
plt.show()

############### IMPLEMENTACION DE REGRESION POLINOMIMAL ###############

from sklearn.model_selection import train_test_split

#Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
x_train, x_test, y_train, y_test = train_test_split(x_p, y_p, test_size=0.2)

from sklearn.preprocessing import PolynomialFeatures

#Se define el grado del polinomio
poli_reg = PolynomialFeatures(degree = 2)

#Se transforma las caracteristicas existentes en caracteristicas de mayor grado
x_train_poli = poli_reg.fit_transform(x_train)
x_test_poli = poli_reg.fit_transform(x_test)

#Definimos el algoritmo a utilizar
pr = linear_model.LinearRegression()

#Entrenamos el modelo
pr.fit(x_train_poli, y_train)

#Realizo una prediccion
y_pred_pr = pr.predict(x_test_poli)

#Graficamos los datos junto con el modelo
plt.scatter(x_test, y_test)
plt.plot(x_test, y_pred_pr, color='red', linewidth=3)
plt.show()

print()
print('DATOS DEL MODELO DE REGRESION POLINOMIAL')
print()

print('Valor de las pendientes o coeficientes "a":')
print(pr.coef_)
print()

print('Valor de la interseccion u ordenada "b":')
print(pr.intercept_)
print()

print('La ecuacion del modelo es iugal a:')
print('y = ', pr.coef_, 'x = ', pr.intercept_)
print()

print('Precision del modelo:')
print(pr.score(x_train_poli, y_train))


































