#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[6]:


# Loading the dataset to a Pandas DataFrame
credit_card_data = pd.read_csv('creditcard.csv')


# In[8]:


# first 5 rows of the dataset
credit_card_data.head()


# In[10]:


credit_card_data.tail()


# In[12]:


# dataset information
credit_card_data.info()


# In[14]:


# checking the number of missing values in each column
credit_card_data.isnull().sum()


# In[18]:


# Distribution of lefit transactions & fraudulent transactions
credit_card_data['Class'].value_counts()


# In[30]:


This Dataset is highly unbalance
0 -> Normal Transaction
1 -> Fraudulent Transaction


# In[23]:


# Separating the data for analysis
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]


# In[25]:


print(legit.shape)
print(fraud.shape)


# In[29]:


# Statistical measures of the data
legit.Amount.describe()


# In[31]:


fraud.Amount.describe()


# In[32]:


# compare the valus for bath transactions
credit_card_data.groupby('Class').mean()


# In[34]:


# Build a sample dataset containing similar distribution of normal transactions and Fraudulent Transactiosn
# Number of Fraudulent Transactions -> 492


# In[37]:


legit_sample = legit.sample(n=492)


# In[38]:


# Concatenating two DataFrames


# In[41]:


new_dataset = pd.concat([legit_sample, fraud], axis=0)


# In[44]:


new_dataset.head()


# In[46]:


new_dataset.tail()


# In[48]:


new_dataset['Class'].value_counts()


# In[51]:


new_dataset.groupby('Class').mean()


# In[55]:


# Splitting the data into Features & Target
X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']


# In[57]:


print(X)


# In[60]:


print(Y)


# In[63]:


# Split the data into Training data and Testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)


# In[65]:


print(X.shape, X_train.shape, X_test.shape)


# In[67]:


print(Y.shape, Y_train.shape, Y_test.shape)


# In[70]:


# MODEL TRAINING 
# Logistic Regression
model = LogisticRegression()


# In[72]:


# training the Logistic Regression Model with Training Data
model.fit(X_train, Y_train)


# In[74]:


# Model Evaluation
# Accuracy Score
# Accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score( X_train_prediction, Y_train)


# In[81]:


print('Accuratcy on Training data:', training_data_accuracy)


# In[84]:


# Accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[87]:


print('Accuracy score on Test data:', test_data_accuracy)

