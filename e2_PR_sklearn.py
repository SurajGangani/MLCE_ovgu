#!/usr/bin/env python
# coding: utf-8

# In[44]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


# ### Import data and split individual properties

# In[45]:


data = pd.read_csv('DATA_superheated_vapor.csv')
data.head()


# In[48]:


V_data = data.loc[ data['Property'] == 'V' ]
V_data.head()


# In[ ]:





# In[50]:


# Get input and output data in seperate column

P = V_data['Pressure']
V = V_data['Liq_Sat']

P = np.array(P).reshape(-1,1)
V = np.array(V).reshape(-1,1)

plt.figure()
plt.plot(P,V, 'kx', markersize=3)                   # plot training data
plt.xlabel('Pressure [kPa]')
plt.ylabel('Specific Volume [$ cm^{-3} g^{-1} $]')
plt.title('Saturated Vapour')
plt.show()


# In[ ]:





# In[ ]:





# ### Polynomial regression: sklearn 
# 
# ###################################################

# In[59]:


# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model  import LinearRegression
# from sklearn.pipeline      import Pipeline

# LR = LinearRegression()
# ss = StandardScaler()
# pf = PolynomialFeatures(degree=5, include_bias=False)

# Pipeline() => Help to arrange events in sequence of operation
# Pipeline([ ('pf', pf), ('ss', ss), ('LR', LR) ])  
#            => First create PolynomialFeatues => Second do StandardizeScaling => Third do LinearRegression


# PR_sklearn = Pipeline([ ('pf', pf), ('ss', ss), ('LR', LR) ])   =>  to generate model 
# PR_sklearn.fit(P, V)                                            =>  to generate model 

# PR_sklearn.predict()                                            =>  to use model and get new output 


# In[61]:


LR = LinearRegression()
ss = StandardScaler()
pf = PolynomialFeatures(degree=5, include_bias=False)

PR_sklearn = Pipeline([ ('pf', pf), ('ss', ss), ('LR', LR) ])
PR_sklearn.fit(P, V)

# create new input
ep = np.linspace(0, P[-1], 100)
y = PR_sklearn.predict(ep)

# plot training data and PR
plt.figure()
plt.xlabel('Pressure [kPa]')
plt.ylabel('Specific Volume [$ cm^{-3} g^{-1} $]')
plt.title('Saturated Liquid  (PR : sklearn)')
plt.plot(P, V, 'kx', markersize=3)
plt.plot(ep, y, 'r')


# In[ ]:





# In[ ]:




