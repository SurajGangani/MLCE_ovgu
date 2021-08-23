#!/usr/bin/env python
# coding: utf-8

# In[125]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[126]:


data = pd.read_csv('DATA_superheated_vapor.csv')
data.head()


# In[127]:


V_data = data.loc[ data['Property'] == 'V' ]
V_data.head()


# In[128]:


# Get inpot and output data in seperate column

P = V_data['Pressure']    # i/o = X (feature)
V = V_data['Liq_Sat']     # o/p = y 

plt.figure() 
plt.plot(P, V, 'kx', markersize = 3)              # plot training data
plt.axvline(300, color = 'r', linestyle = '--')   # plot verticle line at x = 300
plt.axvline(1500, color = 'r', linestyle = '--')  # plot verticle line at x = 1500
plt.xlabel('Pressure [kPa]')
plt.ylabel('Specific Volume [$ cm^{-3} g^{-1} $]')  
plt.title('Saturated Liquid')
plt.show()


# In[150]:


# Extra Step : Divide i/p into 3 section

First_P  = P.loc[ P < 300 ]
Second_P = P.loc[ (300 <= P) & (P < 1500) ]
Third_P  = P.loc[ 1500 <= P ]

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


# In[151]:


# add 1's column in starting of features

First_X  = np.c_[np.ones((First_P.shape[0])), First_P]
Second_X = np.c_[np.ones((Second_P.shape[0])), Second_P]
Third_X  = np.c_[np.ones((Third_P.shape[0])), Third_P]


# In[152]:


# find exact_theta using normal equation

exact_theta_1 = np.linalg.inv(First_X.T.dot(First_X)).dot(First_X.T).dot(First_V)
exact_theta_2 = np.linalg.inv(Second_X.T.dot(Second_X)).dot(Second_X.T).dot(Second_V)
exact_theta_3 = np.linalg.inv(Third_X.T.dot(Third_X)).dot(Third_X.T).dot(Third_V)


# In[153]:


# def line() function

def line(x, theta):
    return theta[0] + theta[1]*x


# In[154]:


# create new i/p

ep_1 = np.linspace(0, 1000, 100)
ep_2 = np.linspace(300, 3000, 100)
ep_3 = np.linspace(2000, 11500, 100)


# In[155]:


# apply new i/p to model and get new o/p 

y_1 = line(ep_1, exact_theta_1)
y_2 = line(ep_2, exact_theta_2)
y_3 = line(ep_3, exact_theta_3)


# In[156]:


# plot given i/p and o/p 
# in same graph also plot generated new i/p and o/p 
# Check how our new generated data using model, fits the given training data

plt.figure()
plt.xlabel('Pressure [kPa]')
plt.ylabel('Specific Volume [$ cm^{-3} g{-1} $]')
plt.title('Saturated Liquid')

plt.plot(First_P, First_V, 'rx', markersize=3)     # plot first section of training data
plt.plot(Second_P, Second_V, 'bx', markersize=3)   # plot second section of training data
plt.plot(Third_P, Third_V, 'kx', markersize=3)     # plot third section of training data

plt.plot(ep_1, y_1, 'r')  # plot new data of first section
plt.plot(ep_2, y_2, 'b')  # plot new data of second section
plt.plot(ep_3, y_3, 'k')  # plot new data of third section

plt.show()


# In[ ]:





# In[ ]:




