# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 19:14:25 2019

@author: hagai
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset=pd.read_csv('Churn_Modelling.csv')
X=dataset.iloc[:, 3:13].values
Y=dataset.iloc[:,13].values
#Encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X1= LabelEncoder()
X[:,1]=labelencoder_X1.fit_transform(X[:,1])
labelencoder_X2= LabelEncoder()
X[:,2]=labelencoder_X2.fit_transform(X[:,2])
onehotencoder=OneHotEncoder(categorical_features= [1])#Use ColumnTranfer Instead
X=onehotencoder.fit_transform(X).toarray()# transform the strings to ints

#X is the dataset in number form
from sklearn.model_selection import train_test_split
#sPLITTING DATASET INTO THE TRAINGING AND TEST
X_train, X_test, y_train, y_test =train_test_split(X,Y,test_size= 0.25,random_state=0)

#sCALING
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
X_train=sc.fit_transform(X_train)
#X_test ,if each costumer left
X_test = sc.transform(X_test)#Scaling easies calculations

#Creating the classifier

import keras
from keras.models import Sequential
from keras.layers import Dense# Layers for the Artificial Network

#Initialize the Artificial Neuron Network
classifier = Sequential()#Sequence of layers
#Dense is a layer,we are adding hidden layers to the ANN sequence.
classifier.add(Dense(activation ='relu',units=6, kernel_initializer= 'uniform',input_dim = 12))
classifier.add(Dense(activation ='relu',units=6, kernel_initializer= 'uniform'))

#output layer 
classifier.add(Dense(activation ='sigmoid',units=1, kernel_initializer= 'uniform'))

classifier.compile(optimizer= 'adam', loss= 'binary_crossentropy',metrics =['accuracy'])
classifier.fit(X_train, y_train,batch_size = 10, epochs=100)



#Rectfier for Hidden,Siegmund for Output layer.

#prepare the prediction
y_pred=classifier.predict(X_test)#Probability of leaving
y_pred=(y_pred>0.5)#separates into leaving and not leaving

#Predicting a single new observations

new_prediction = classifier.predict(sc.transform(np.array([[0,0,0,600,1,40,3,60000,2,1,1,50000]])))

new_prediction =(new_prediction >0.5)

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test, y_pred)