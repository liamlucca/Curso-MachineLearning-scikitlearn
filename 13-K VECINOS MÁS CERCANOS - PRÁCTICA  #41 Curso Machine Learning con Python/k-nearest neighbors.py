#K VECINOS MAS CERCANOS

############### PREPARAR LA DATA ###############
from sklearn import datasets

#Importamos los datos de la misma libreria de scikit-learn
dataset = datasets.load_breast_cancer()
print(dataset)
print()

############### ENTENDIMIENTO DE LA DATA ###############

#Verifico la informacion contenida en el dataset
print('Informacion en el dataset:')
print(dataset.keys())
print()

#Verifico las caracteristicas del dataset
print('Caracteristicas del dataset:')
print(dataset.DESCR)
print()

#Verifico la cantidad de datos que hay en los dataset
print('Cantidad de datos:')
print(dataset.data.shape)
print()

#Verifico la informacion de las columnas
print(dataset.feature_names)
print()
    
############### PREPARAR LA DATA ###############

#Seleccionamos todas las columnas
x = dataset.data

#Definimos los datos correspondientes a las etiquetas
y = dataset.target

############### IMPLEMENTACION DE K VECINOS MAS CERCANOS ###############

from sklearn.model_selection import train_test_split

#Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

#Definimos el algoritmo a utilizar
from sklearn.neighbors import KNeighborsClassifier
algoritmo = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

#Entrenamos el modelo
algoritmo.fit(x_train, y_train)

#Realizo una prediccion
y_pred = algoritmo.predict(x_test)

############### METRICAS DE EVALUACION ###############

print('DATOS DEL MODELO DE K-VECINOS MAS CERCANOS')
print()

#Verifico la matriz de confusion
from sklearn.metrics import confusion_matrix
matriz = confusion_matrix(y_test, y_pred)
print('Matriz de Confusion')
print(matriz)
print()

#Calculo la presicion del modelo
from sklearn.metrics import precision_score
precision = precision_score(y_test, y_pred)
print('Precision')
print(precision)
print()

#Calculo la exactitud del modelo
from sklearn.metrics import accuracy_score
exactitud = accuracy_score(y_test, y_pred)
print('Exactitud del modelo')
print(exactitud)
print()

#Calculo la sensibilidad del modelo
from sklearn.metrics import recall_score
sensibilidad = recall_score(y_test, y_pred)
print('Sensibilidad del modelo')
print(sensibilidad)
print()

#Calculo el puntaje F1 del modelo
from sklearn.metrics import f1_score
puntajef1 = f1_score(y_test, y_pred)
print('Puntaje F1 del modelo')
print(puntajef1)
print()

#Calculo la curva ROC = AUC del modelo
from sklearn.metrics import roc_auc_score
roc_auc = roc_auc_score(y_test, y_pred)
print('Curva ROC - AUC del modelo')
print(roc_auc)