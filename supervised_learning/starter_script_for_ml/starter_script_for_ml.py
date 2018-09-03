#importing libraries
import numpy as n
import matplotlib.pyplot as m
import pandas as p

#loading the dataset
dataset = p.read_csv('dataset.csv', sep=',').values

#filling blank cells
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis = 0)
imputer = imputer.fit(dataset[:, 2:6])
dataset[:, 2:6] = imputer.transform(dataset[:, 2:6])

#turning textual data to numerical
from sklearn.preprocessing import LabelEncoder
labelencoder_0 = LabelEncoder() #independent variable encoder
dataset[:,0] = labelencoder_0.fit_transform(dataset[:,0])
labelencoder_1 = LabelEncoder() #independent variable encoder
dataset[:,1] = labelencoder_1.fit_transform(dataset[:,1])
labelencoder_6 = LabelEncoder() #dependent (target) variable encoder
dataset[:,6] = labelencoder_6.fit_transform(dataset[:,6])

#taking care of wrong order relationships
from sklearn.preprocessing import OneHotEncoder
onehotencoder_01 = OneHotEncoder(categorical_features = [0, 1])
dataset = onehotencoder_01.fit_transform(dataset).toarray()

#splitting the dataset into the source variables (independant variables) and the target variable (dependant variable)
sourcevars = dataset[:,:-1] #all columns except the last one
targetvar = dataset[:,len(dataset[0])-1] #only the last column

#feature scaling
from sklearn.preprocessing import StandardScaler
stScaler_ds = StandardScaler()
sourcevars = stScaler_ds.fit_transform(sourcevars)

#splitting the dataset into training and test set
from sklearn.cross_validation import train_test_split
sv_train, sv_test, tv_train, tv_test = train_test_split(sourcevars, targetvar, test_size=0.2, random_state=0)
