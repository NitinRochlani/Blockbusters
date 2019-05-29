# -*- coding: utf-8 -*-
"""
Created on Wed May 22 23:06:53 2019

@author: Raghad
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

#Importing data
train_df = pd.read_csv('../Data/train.csv')
test_df = pd.read_csv('../Data/test.csv')

#Preparing data
train_X_df = prepping_data(train_df)
train_y_df = train_df[['id','revenue']].set_index('id')

test_X_df = prepping_data(test_df)

#EDA
print(train_df.describe())
print(train_df.info())
#Revenue EDA
print("----Revenue Stats----")
print(train_df['revenue'].describe())

f, ax = plt.subplots(figsize=(15, 8.5))
rev_g = sns.distplot(train_df['revenue'],bins=50, ax= ax)
plt.show()
#Budget EDA
print("----Budget Stats----")
print(train_df['budget'].describe())

#Revenue & Budget Correlation
bud_g = sns.regplot(x=train_df['budget'], y=train_df['revenue'])
plt.show()
#Collections EDA
#New column 'has_collection
print("----Belongs_to_collection Graphs----")
train_df['has_collection'] = train_df['belongs_to_collection'].apply(lambda x: 0 if pd.isna(x) else 1)
col_g = sns.countplot(x='has_collection', data=train_df)
plt.show()
f, ax = plt.subplots(figsize=(10, 15))
col_g1 = sns.boxplot(x='has_collection', y='revenue', data=train_df, ax= ax)
plt.show()

#Runtime EDA
print('----Runtime Stats----')
print(train_df['runtime'].describe())
train_df['runtime'] = train_df['runtime'].apply(lambda x: 0 if pd.isna(x) else x)
f, ax = plt.subplots(figsize=(10, 15))
runtime1 = sns.distplot(train_df['runtime'],bins=150, ax=ax)
plt.show()
