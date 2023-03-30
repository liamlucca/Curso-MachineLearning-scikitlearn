# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 14:59:08 2021

@author: liaml
"""

import numpy as np
import pandas as pd

# Series

series = pd.Series({"Argentina":"Buenos Aires",
                    "Chile":"Santiago de Chile",
                    "Colombia":"Bogota",
                    "Peru":"Lima"})

print('Series:')
print(series)
print()

# Dataframe

data = np.array([['','Col1','Col2'], ['Fila1',11,12], ['Fila2', 33, 44]])
df = pd.DataFrame(data=data[1:,1:], index=data[1:,0], columns=data[0,1:])
print('DataFrame:')
print(df)
print()

# Forma del data frame

print('Forma del DataFrame')
print(df.shape)
print()

# Altura del DataFrame

print('Altura del DataFrame:')
print(len(df.index))
print()

# Estadisticas del Dataframe

print('Estadisticas del DataFrame:')
print(df.describe())
print()

# Media de las columnas DataFrame

print('Media de las columnas DataFrame:')
print(df.mean())
print()

# Correlacion del DataFrame

print('Correlacion del DataFrame:')
print(df.corr())
print()

# Cuenta los datos del DataFrame

print('Conteo de datos del DataFrame:')
print(df.count())
print()

# Valor mas alto de cada columna del DataFrame

print('Valor más alto de la columna del DataFrame:')
print(df.max())
print()

# Valor mas bajo de cada columna del DataFrame

print('Valor más bajo de la columna del DataFrame:')
print(df.min())
print()

# Mediana de cada columna

print('Mediana de la columna del DataFrame:')
print(df.median())
print()

# Desviacion estandar de cada columna del DataFrame

print('Desviacion estandar de cada columna del DataFrame:')
print(df.std())
print()

# Seleccionar la primera columna del DataFrame

#print('Primera columna del DataFrame:')
#print(df[0])


# Seleccionar dos columnas del Data Frame

#print('Seleccionar dos columnas del Data Frame:')
#print(df[[0, 1]])

# Seleccionar el valor de la primera fila y ultima columna del DataFrame

print('Valor de la primera fila y ultima columna del DataFrame:')
print(df.iloc[0][1])
print()

# Seleccionar los valores de la primera fila del DataFrame

print('Valores de la primera fila del DataFrame:')
print(df.iloc[0])
print()

# Seleccionar los valores de la primera fila del DataFrame

print('Valores de la primera fila del DataFrame:')
print(df.iloc[0, :])
print()

#Verificar si hay datos nulos en el DataFrame

print('Datos nulos en el DataFrame:')
print(df.isnull())
print()

#Suma de datos nulos en el DataFrame

print('Datos nulos en el DataFrame:')
print(df.isnull().sum())
print()

#Remplaza los valores perdidos por la media

print('Remplaza los valores perdidos por la media')
print(df.fillna(df.mean()))











 
