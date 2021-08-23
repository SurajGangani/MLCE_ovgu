#!/usr/bin/env python
# coding: utf-8

# In[227]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler


# In[228]:


data = pd.read_csv('DATA_superheated_vapor.csv')
data.head()


# In[230]:


V_data = data.loc[ data['Property'] == 'V' ]
U_data = data.loc[ data['Property'] == 'U' ]
H_data = data.loc[ data['Property'] == 'H' ]
S_data = data.loc[ data['Property'] == 'S' ]


# In[ ]:





# In[ ]:





# # Polynomial regression
# 
# ###################################################

# #### General rules for PR

# In[224]:


# NAMING IN MACHINE LEARNING
#   given data = training data
#  given input = feature
# given output = target / label

# training data is seperated in inputs and output(if given)

#            X = (x1, x2, x3, .....)  =>  known as features (inputs)
# In reality X = (x0, x1, x2, .....)  =>  X also contains x0, but it is always column of 1 and added later


####------- STEPS TO DO PR (normal equation) -------####
# 1) get input X and output y as seperate column from given data
# 2) create new features
# 3) add 1's column in starting of given i/p and rename it to X 
# 4) find 'norm_theta' using normal equation (given in lec slide)
# 5) def 'polynomial' fun using calculated 'norm_theta'   
#
#      °°°°°  THIS 'polynomial' FUN IS OUR CREATED MODEL FROM GIVEN TRAINING DATA  
#      °°°°°  NOW WE CAN USE THIS MODEL TO GENERATE MORE DATA; SIMILAR TO GIVEN TRAINING 
#
# 6) create any new inputs
# 7) apply new inputs to our model (polynomial fun) and store new output
# 8) plot given i/p and o/p 
# 9) in same graph also plot generated new i/p and o/p 
# 10) Check how our new generated data using model, fits the given training data


# In[ ]:





# In[ ]:





# In[210]:


# 1) get input X and output y as seperate column from given data

P = V_data['Pressure']
P = np.array(P).reshape(-1,1)

V = V_data['Liq_Sat']


# In[211]:


# 2) create new features

feature_1 = P**1
feature_2 = P**2
feature_3 = P**3
feature_4 = P**4
feature_5 = P**5

Features = np.concatenate((feature_1, feature_2, feature_3, feature_4, feature_5), axis=1)


# In[213]:


# 3) add 1's column in starting of given i/p and rename it to X 

FeaturesOne = np.append(np.ones((Features.shape[0],1)),Features, axis=1)


# In[214]:


# 4) find 'norm_theta' using normal equation (given in lec slide)

X = FeaturesOne
norm_theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(V)


# In[216]:


# 5) def 'polynomial' fun using calculated 'norm_theta'  

def polynomial(x, theta):
    return theta[0] + theta[1]*(x**1) + theta[2]*(x**2) + theta[3]*(x**3) + theta[4]*(x**4) + theta[5]*(x**5)


# In[241]:


# 6) create any new inputs

ep = np.linspace(0,P[-1], 100)


# In[242]:


# 7) apply new inputs to our model (polynomial fun) and store new output

y = polynomial(ep, norm_theta)


# In[243]:


# 8) plot given i/p and o/p 
# 9) in same graph also plot generated new i/p and o/p 
# 10) Check how our new generated data using model, fits the given training data

plt.figure()
plt.plot(P, V, 'kx', markersize=3)  # training data
plt.plot(ep, y, 'r')   # new data (PR Model)
plt.xlabel('Pressure [kPa]')
plt.ylabel('Specific Volume [$ cm^{-3} g{-1}$]')
plt.title('Saturated Liquid', size=15)
plt.show()


# In[ ]:





# In[ ]:




