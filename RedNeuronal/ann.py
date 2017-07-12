#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 08:12:00 2017

@author: fabian
"""

#STEP 1> PREPROCESAMIENTO

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
#mostrar el contenido del array
np.set_printoptions(threshold=np.nan)

#importing dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

#codificar datos categ'oricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1country = LabelEncoder()
X[:, 1] = labelencoder_X_1country.fit_transform(X[:, 1])
labelencoder_gender = LabelEncoder()
X[:, 2] = labelencoder_gender.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
#dummy variable trap
X = X[:, 1:]

#hacer la divisi'on de nuestro datset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.2,random_state = 0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


'''
#STEP 2: VAMOS A CREAR NUESTRA RED NEURONAL
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

#inicializar ANN
#Agregar la capa input y luego la capa hidden
classifier = Sequential()

classifier.add(Dense(input_shape=(11,), units=6, kernel_initializer='uniform', activation='relu'))
classifier.add(Dropout(rate=0.1))

#agregar segunda capa
classifier.add(Dense(units=6, kernel_initializer='uniform',activation='relu'))
classifier.add(Dropout(rate=0.1))

#AGREGAR LA CAPA OUTPUT
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
#si tuvi'eramos m'as neuronas de salida, no se usa sigmoid
#en ese caso se usa softmax

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#CORRER EL MODELO
classifier.fit(X_train, y_train, batch_size=5, epochs=20)



#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

"""
Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $60000
Number of Products: 2
Does this customer have a credit card ? Yes
Is this customer an Active Member: Yes
Estimated Salary: $50000
"""

sample = np.array([(
        0,0,
        600.0,
        1,
        40,
        3,
        60000,
        2,
        1,
        1,
        50000
        )])

sample = sc.transform(sample)
myPrediction = classifier.predict(sample)
myPrediction = (myPrediction > 0.5)


#CROSS VALIDATION
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

import keras
from keras.models import Sequential
from keras.layers import Dense

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(input_shape=(11,), units=6, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return classifier

classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, epochs=10)
#@VARIABLES: cv-> size of the folds(remember fold are the slide sizes of the training set). n_jobs-> cpu's used for execution
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10, n_jobs=1)

mean = accuracies.mean()
variance = accuracies.std()


'''

#Hiperparametros
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

from keras.models import Sequential
from keras.layers import Dense

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(input_shape=(11,), units=6, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=50, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return classifier

classifier = KerasClassifier(build_fn=build_classifier)

parameters = {'batch_size': [15],
               'epochs': [400],
               'optimizer':['adam', 'rmsprop','adadelta']}

grid_search = GridSearchCV(estimator=classifier,
                           param_grid=parameters,
                           scoring='accuracy',
                           cv=10)

grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

print best_parameters
print best_accuracy
