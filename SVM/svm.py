# STEP 1> PREPROCESAMIENTO
from sklearn.svm.tests.test_svm import test_libsvm_parameters

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# mostrar el contenido del array
np.set_printoptions(threshold=np.nan)

# importing dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# codificar datos categ'oricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_1country = LabelEncoder()
X[:, 1] = labelencoder_X_1country.fit_transform(X[:, 1])
labelencoder_gender = LabelEncoder()
X[:, 2] = labelencoder_gender.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
# dummy variable trap
X = X[:, 1:]

# hacer la divisi'on de nuestro datset
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# feature scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


#maquina de soporte vectorial
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, recall_score,precision_score, classification_report, confusion_matrix

model = SVC(kernel='rbf').fit(X_train, y_train)

# Predicting the Test set results
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

print 'Accuracy:', accuracy_score(y_test, y_pred)
print 'F1 score:', f1_score(y_test, y_pred,average='weighted')
print 'Recall:', recall_score(y_test, y_pred,average='weighted')
print 'Precision:', precision_score(y_test, y_pred,average='weighted')
print '\n clasification report:\n', classification_report(y_test,y_pred)
print '\n confussion matrix:\n',confusion_matrix(y_test, y_pred)
