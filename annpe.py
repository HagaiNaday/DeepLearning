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

from sklearn.preprocessing import LabelEconder, OneHotEncoder
labelenconder_X1= LabelEncoder()
X[:,1]=labelencoder_X1.fit_transform(X[:,1])
