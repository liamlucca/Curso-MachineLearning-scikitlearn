# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 14:59:08 2021

@author: liaml
"""

import matplotlib.pyplot as plt


#Definir los datos
x1 = [3, 4, 5, 6]
y1 = [5, 6, 3, 4]
x2 = [2, 5, 8]
y2 = [3, 4, 3]


    #Diagrama de Linea

#Configurar las caracteristicas del grafico
plt.plot(x1, y1, label = "Linea 1", linewidth = 4, color = 'blue')
plt.plot(x2, y2, label = "Linea 2", linewidth = 4, color = 'green')

#Definir titulo y nombres de ejes
plt.title('Diagrama de Lineas')
plt.xlabel('Eje X')
plt.ylabel('Eje Y')

#Mostrar leyenda, cuadricula y figura
plt.legend()
plt.grid()
plt.show()


    #Diagrama de barras


#Configurar las caracteristicas del grafico
plt.bar(x1, y1, label = "Datos 1", width = 0.5, color = 'blue')
plt.bar(x2, y2, label = "Datos 2", width = 0.5, color = 'green')

#Definir titulo y nombres de ejes
plt.title('Diagrama de Barras')
plt.xlabel('Eje X')
plt.ylabel('Eje Y')

#Mostrar leyenda, cuadricula y figura
plt.legend()
plt.grid()
plt.show()


    #Histograma

#Definir los datos
a = [22,55,62,45,21,40,60,30,20,70,23,45,57,64,63,23,64,20,11,15,16,14]
b = [10,20,30,40,50,60,70,80,90,100]

#Configurar las caracteristicas del grafico
plt.hist(a, b, histtype = "bar", rwidth = 0.8, color = 'lightgreen')

#Definir titulo y nombres de ejes
plt.title('Histogramas')
plt.xlabel('Eje X')
plt.ylabel('Eje Y')

#Mostrar leyenda, cuadricula y figura
plt.legend()
plt.grid()
plt.show()


    #Grafico de dispersion
    
#Configurar las caracteristicas del grafico
plt.scatter(x1, y1, label = 'Datos 1', color = 'red')
plt.scatter(x2, y2, label = 'Datos 2', color = 'purple')

#Definir titulo y nombres de ejes
plt.title('Grafico de dispersion')
plt.xlabel('Eje X')
plt.ylabel('Eje Y')

#Mostrar leyenda, cuadricula y figura
plt.legend()
plt.grid()
plt.show()


    #Grafico circular
    
#Definir los datos
dormir = [7,8,6,11,7]
comer = [2,3,4,3,2]
trabajar = [7,8,7,2,2]
recreacion = [8,5,7,8,13]
divisiones = [7,2,2,13]
actividades = ['Dormir','Comer','Trabajar','Recreacion']
colores = ['red','purple','blue','orange']
    
#Configurar las caracteristicas del grafico
plt.pie(divisiones, labels=actividades, colors=colores, startangle=90, 
        shadow=True, explode=(0.1,0,0,0), autopct='%1.1f%%')

#Definir titulo y nombres de ejes
plt.title('Grafico circular')

#Mostrar leyenda, cuadricula y figura
plt.show()

