#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression


# In[3]:


data = pd.read_csv('DATA_superheated_vapor.csv')
data.head()


# In[4]:


V_data = data.loc[ data['Property'] == 'V' ]
V_data.head()


# In[5]:


# Get input and output data in seperate column

P = V_data['Pressure']   # i/o = X (feature)
V = V_data['Liq_Sat']    # o/p = y 

plt.figure()
plt.plot(P,V, 'kx', markersize=3)                   # plot training data
plt.axvline(300, color = 'r', linestyle = '--')     # plot verticle line at x = 300
plt.axvline(1500, color = 'r', linestyle = '--')    # plot verticle line at x = 1500
plt.xlabel('Pressure [kPa]')
plt.ylabel('Specific Volume [$ cm^{-3} g^{-1} $]')
plt.title('Saturated Vapour')
plt.show()


# In[6]:


# Extra Step : Divide i/p into 3 section

First_P  = P.loc[ P < 300]
Second_P = P.loc[ (300 <= P) & (P < 1500) ]
Third_P  = P.loc[ 1500 <= P]

First_V  = V.loc[ P < 300 ]
Second_V = V.loc[ (300 <= P) & (P < 1500) ]
Third_V  = V.loc[ 1500 <= P ]

# convert in np.array()
First_P  = np.array(First_P).reshape(-1, 1)
Second_P = np.array(Second_P).reshape(-1, 1)
Third_P  = np.array(Third_P).reshape(-1, 1)
First_V  = np.array(First_V).reshape(-1, 1)
Second_V = np.array(Second_V).reshape(-1, 1)
Third_V  = np.array(Third_V).reshape(-1, 1)


# In[ ]:





# In[ ]:





# ### Linear regression: sklearn 
# 
# ###################################################

# In[7]:


# from sklearn.linear_model import LinearRegression

#  model_name = LinearRegression().fit( training_input, training_output )  =>  fits the training input to training output
#  model_name.predict( new_input )                                         =>  predict new_output from new_input


#  LR = LinearRegression()
#  LR.fit()      =>  to generate model 
#  LR.predict()  =>  to use model and get new output 


# In[9]:


# create new example for i/p
 
ep_1 = np.linspace(0, 1000, 100).reshape(-1,1)
ep_2 = np.linspace(300, 3000, 100).reshape(-1,1)
ep_3 = np.linspace(2000, 11500, 100).reshape(-1,1)


# In[10]:


# generate model using training data and fit to new data

# ALWAYS PREDICT OUTPUT FOR NEW DATA, IMMEDIATLY AFTER GENERATING MODEL FROM TRAINING DATA

LR = LinearRegression()

LR_sklearn_1 = LR.fit(First_P, First_V)    # fit first training data to generate model
y_1 = LR_sklearn_1.predict(ep_1)           # predict output for new input 
LR_sklearn_2 = LR.fit(Second_P, Second_V)  # fit second training data to generate model
y_2 = LR_sklearn_2.predict(ep_2)           # predict output for new input 
LR_sklearn_3 = LR.fit(Third_P, Third_V)    # fit third training data to generate model
y_3 = LR_sklearn_3.predict(ep_3)           # predict output for new input 


# In[11]:


# plot training data and in same graph plot new data

plt.figure()
plt.xlabel('Pressure [kPA]')
plt.ylabel('Specific Volume [$ cm^{-3} g^{-1} $]')  
plt.title('Saturated Liquid (sklearn)')

plt.plot(First_P, First_V, 'rx', markersize=2, label='First split data')     # training data
plt.plot(Second_P, Second_V, 'bx', markersize=2, label='Second split data')  # training data
plt.plot(Third_P, Third_V, 'kx', markersize=2, label='First split data')     # training data

plt.plot(ep_1, y_1, 'r')
plt.plot(ep_2, y_2, 'b')
plt.plot(ep_3, y_3, 'k')

plt.show()


# In[ ]:





# In[ ]:




