#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 19:30:49 2019

@author: raghad
Based on the template of Alexander Melde
https://www.kaggle.com/alexandermelde/code-template-for-simple-regression-prediction
"""
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
#from sklearn import linear_model
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split

#Preprocessing Methods
def prepping_data(df):
    #sets 'id' attribute as index
    df = df.set_index('id')

    #Extracts the easiest attributes to process
    df = df[['budget','original_language','popularity','runtime', 'status']]

    #Transforms non-numeric values to binary columns
    df = pd.get_dummies(df)
    
    #removes NA values
    return df.fillna(0)

def evaluating_score(true_y, pred_y):
    
#    pred_y = np.where(pred_y >0, pred_y, 0)
#    
#    return np.sqrt(mean_squared_log_error(true_y, pred_y))
    assert len(true_y) == len(pred_y)
    return np.sqrt(np.mean((np.log1p(pred_y) - np.log1p(true_y))**2))

def inverseY(y):
    return np.exp(scaler_y.inverse_transform(np.reshape(y, (-1,1))))

#Importing data
train_df = pd.read_csv('../Data/train.csv')
test_df = pd.read_csv('../Data/test.csv')

#Preparing data
train_X_df = prepping_data(train_df)
train_y_df = train_df[['id','revenue']].set_index('id')

test_X_df = prepping_data(test_df)

#Add collections col
train_df = train_df.set_index('id')
test_df = test_df.set_index('id')

train_X_df['has_collection'] = train_df['belongs_to_collection'].apply(lambda x: 0 if pd.isna(x) else 1)
test_X_df['has_collection'] = test_df['belongs_to_collection'].apply(lambda x: 0 if pd.isna(x) else 1)

train_X_df['has_homepage'] = train_df['homepage'].apply(lambda x: 0 if pd.isna(x) else 1)
test_X_df['has_homepage'] = test_df['homepage'].apply(lambda x: 0 if pd.isna(x) else 1)


#Ensuring that both sets have the same dummy variables
train_X_df, test_X_df = train_X_df.align(test_X_df, join='outer', axis=1, fill_value=0)

#Don't know why, but Alex removes metadata
train_X = train_X_df.values
train_y = train_y_df.values
test_X = test_X_df.values

#Validation Split
train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=0.7, random_state=56)

#Scale and Normalize
scaler_X = StandardScaler()

train_X_scaled = scaler_X.fit_transform(train_X)
val_X_scaled = scaler_X.transform(val_X)

test_X_scaled = scaler_X.transform(test_X)
#train_y = np.log(train_y)
scaler_y = MinMaxScaler((0,1))
train_y_scaled = scaler_y.fit_transform(np.log(train_y)).ravel()# transform and convert column-vector y to a 1d array


#Training Model and producing R2 coefficents
lm = KNeighborsRegressor().fit(train_X_scaled,train_y_scaled)
#lm = linear_model.LinearRegression()
#model = lm.fit(train_X,train_y)

#Apply model 

y_train_pred  = inverseY(lm.predict(train_X_scaled))
y_val_pred    = inverseY(lm.predict(val_X_scaled))
y_test_pred   = inverseY(lm.predict(test_X_scaled))


print("RMLS Error on Training Dataset:", evaluating_score(train_y , y_train_pred))
print("RMLS Error on Val Dataset:", evaluating_score(val_y , y_val_pred))


#Submssion file
test_df = test_X_df.assign(revenue=y_test_pred)

test_y_df = test_df[['revenue']]

test_y_df.to_csv('submission.csv')
