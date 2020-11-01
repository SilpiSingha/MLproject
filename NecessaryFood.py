#importing the libraries
#for doing mathematical/arithmetic computation
import numpy as np

#for importing data from datasets
import pandas as pd

#for visualization of results
import matplotlib.pyplot as plt

#Step-1
#reading the data from dataset
dataset = pd.read_csv('Necessary_Food.csv')

#extracting the independent variable
X = dataset.iloc[0:,0:4].values

#extracting the dependent variable
y = dataset.iloc[0:,5:6].values

#Step-2
#removal of missing data
from sklearn.preprocessing import Imputer
impute = Imputer(missing_values="NaN", strategy="mean",axis = 0)

#trains the machine on the data
impute.fit(X[:,1:3])

#applying the training on dataset
X[:,1:3] = impute.transform(X[:,1:3])

#Step-3
#Removing Categorical Data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
lab_enc = LabelEncoder()
X[:,0] = lab_enc.fit_transform(X[:,0])
'''oneHot = OneHotEncoder(categorical_features=[0])
X = oneHot.fit_transform(X).toarray()'''

'''lab_encY = LabelEncoder()
y = lab_encY.fit_transform(y)'''

'''#Step-4
#applying feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)'''


#Step-5
#splitting the dataset into test and train
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.85, random_state=0)



'''
#implement the model for simple linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#predicting the outcomes based on the 
#trained model
y_pred = regressor.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score
lin_mse= mean_squared_error(y_test,y_pred)
lin_r2s= r2_score(y_test,y_pred)

lin_act = sum(y_test)/len(y_test)
lin_obt = sum(y_pred)/len(y_pred)

perc_acc_lin = 1-((lin_act-lin_obt)/lin_act)*100


#training the model with the dataset for Multiple Linear regression
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train,y_train)

#predicting the outcomes on test data
M_y_pred = regression.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score
Mlin_mse= mean_squared_error(y_test,M_y_pred)
Mlin_r2s= r2_score(y_test,M_y_pred)


Mlin_act = sum(y_test)/len(y_test)
Mlin_obt = sum(My_pred)/len(My_pred)

perc_acc_Mlin = 1-((Mlin_act-Mlin_obt)/Mlin_act)*100




#fit the model in multiple decision tree
from sklearn.tree import DecisionTreeRegressor
reg_dt = DecisionTreeRegressor(random_state=0)
reg_dt.fit(X_train,y_train)

#predicting the results for decision tree 
y_pred_dt = reg_dt.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score
Dt_mse= mean_squared_error(y_test,y_pred_dt)
Dt_r2s= r2_score(y_test,y_pred_dt)


dt_act = sum(y_test)/len(y_test)
dt_obt = sum(y_pred_dt)/len(y_pred_dt)

perc_acc_dt = 1-((dt_act-dt_obt)/dt_act)*100'''


#fit the model in Random forest
from sklearn.ensemble import RandomForestRegressor
reg_rf= RandomForestRegressor(n_estimators= 10, random_state= 0)
reg_rf.fit(X_train,y_train)


#predicting the results for Random forest 
y_pred_rf = reg_rf.predict(X_test)

from sklearn.metrics import r2_score
Rf_r2s= r2_score(y_test,y_pred_rf)

#G=reg_rf.predict([[1,174,96,12]])



import pickle
with open('Model_pickle','wb') as file:
    pickle.dump(reg_rf,file)
with open('Model_pickle','rb') as f:
    Obj=pickle.load(f)


