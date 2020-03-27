# -*- coding: utf-8 -*-

import pandas as pd
data = pd.read_csv('datos_proyecto.csv')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

import seaborn as sns

data_m = data.iloc[:,0:7]
etiquetas = data.iloc[:,7]
X_train, X_test, y_train, y_test = train_test_split(data_m, etiquetas, test_size=0.3, random_state=0)

datos = pd.concat([data_m,pd.DataFrame({"abusivo": etiquetas})],axis=1)
sns.pairplot(datos,hue="abusivo",vars=data_m.columns)
#SVM
scaler= StandardScaler()
scaler.fit(X_train)
Z_train = pd.DataFrame(scaler.transform(X_train))
Z_test = pd.DataFrame(scaler.transform(X_test))
model = svm.SVC(kernel = 'rbf')
model.fit(Z_train, y_train)
y_pred = model.predict(Z_test)
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test, y_pred))

#Logistic
model2 = LogisticRegression()
model2.fit(Z_train, y_train)
y_pred2 = model2.predict(Z_test)
print(classification_report(y_test,y_pred2))
print(confusion_matrix(y_test, y_pred2))

#randomforest
from sklearn.ensemble import RandomForestClassifier
model3 = RandomForestClassifier(n_estimators = 1000)
model3.fit(Z_train, y_train)
y_pred3 = model3.predict(Z_test)
print(classification_report(y_test,y_pred3))
print(confusion_matrix(y_test, y_pred3))

#NN
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 7))
classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(Z_train, y_train, batch_size = 10, epochs = 100)
y_pred4 = classifier.predict(Z_test)
y_pred4 = y_pred4.round()
print(classification_report(y_test,y_pred4))
print(confusion_matrix(y_test, y_pred4))

#con k means
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

scaler.fit(data_m)
Z_k = pd.DataFrame(scaler.transform(data_m))
dis = []
K = range(1,10)
for k in K:
    modelok = KMeans(n_clusters=k)
    modelok.fit(Z_k)
    dis.append(modelok.inertia_)

plt.figure(figsize=(16,8))
plt.plot(K, dis, 'bx-')
plt.xlabel('Valores de K')
plt.ylabel('WCSS')
plt.title('Método del codo')
plt.show()

modelok = KMeans(n_clusters = 4)
modelok.fit(Z_k)
y = modelok.labels_
Z_k['cluster'] = y
etiquetas = etiquetas.to_frame()
etiquetas['cluster'] = y
X_train, X_test, y_train, y_test = train_test_split(Z_k, etiquetas, test_size=0.3, random_state=0)
for i in range(0,4):
    print(i)
    print(X_test[X_test['cluster'] == i].count())
    model3 = RandomForestClassifier(n_estimators = 1000)
    model3.fit(X_train[X_train['cluster'] == i].drop(columns = 'cluster'), y_train[y_train['cluster'] == i].iloc[:,0])
    y_pred3 = model3.predict(X_test[X_test['cluster'] == i].drop(columns = 'cluster'))
    print(classification_report(y_test[y_test['cluster'] == i].iloc[:,0],y_pred3))
    print(confusion_matrix(y_test[y_test['cluster'] == i].drop(columns = 'cluster').iloc[:,0], y_pred3))