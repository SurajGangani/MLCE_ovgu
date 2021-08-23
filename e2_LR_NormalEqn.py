#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ### Import data and split individual properties

# In[2]:


data = pd.read_csv('DATA_superheated_vapor.csv')
data.head()


# In[3]:


V_data = data.loc[data['Property'] == 'V']
U_data = data.loc[data['Property'] == 'U']
H_data = data.loc[data['Property'] == 'H']
S_data = data.loc[data['Property'] == 'S']


# In[ ]:





# In[ ]:





# ## Linear regression: Normal Eqn
# 
# ###################################################

# #### General rules for LR

# In[4]:


# NAMING IN MACHINE LEARNING
#   given data = training data
#  given input = feature
# given output = target / label

# training data is seperated in inputs and output(if given)

#            X = (x1, x2, x3, .....)  =>  known as features (inputs)
# In reality X = (x0, x1, x2, .....)  =>  X also contains x0, but it is always column of 1 and added later


####------- STEPS TO DO LR (normal equation) -------####
# 1) get input X and output y as seperate column from given data
# 2) add 1's column in starting of given i/p and rename it to X 
# 3) find 'exact_theta' using normal equation (given in lec slide)
# 4) def 'line' fun using calculated 'exact_theta'   
#
#      °°°°°  THIS 'line' FUN IS OUR CREATED MODEL FROM GIVEN TRAINING DATA  
#      °°°°°  NOW WE CAN USE THIS MODEL TO GENERATE MORE DATA; SIMILAR TO GIVEN TRAINING 
#
# 5) create any new inputs
# 6) apply new inputs to our model (line fun) and store new output
# 7) plot given i/p and o/p 
# 8) in same graph also plot generated new i/p and o/p 
# 9) Check how our new generated data using model, fits the given training data


# #### Applying rules to our case

# In[5]:


# In our case given input  X = Pressure          = P
#            given output  y = volume of Liq_Sat = V


####------- STEPS TO DO LR (normal equation) -------#### 
# 1) get input X and output y as seperate column                     =>   we collected  P, and Liq_Sat = V   [given i/p = P,  given o/p = V]
# 2) add 1's column in starting of given i/p and rename it to X                                  
# 3) find 'exact_theta' using normal equation (given in lec slide)
# 4) def 'line' fun using calculated 'exact_theta'   
#
# 5) create new inputs                                               =>   we create evolution_points  [new i/p = evolution_points]                       
# 6) apply new inputs to our model (line fun) and store new output   =>   we store with name 'y'      [new o/p = y]
# 7) plot given i/p and o/p 
# 8) in same graph also plot generated new i/p and o/p 
# 9) Check how our new generated data using model, fits the given training data


# In[6]:


# 1) get input X and output y as seperate column  =>  we collected  P, and Liq_Sat = V   [given i/p = P,  given o/p = V]

P = V_data['Pressure']
P = np.array(P).reshape(-1,1)

V = V_data['Liq_Sat']
V = np.array(V)

print(P.shape)
print(V.shape)


# In[7]:


# 2) add 1's column in starting of given i/p and rename it to X 

X = np.append(np.ones(P.shape),P, axis=1) 


# In[8]:


# 3) find 'exact_theta' using normal equation (given in lec slide)

exact_theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(V)


# In[23]:


# 4) def 'line' fun using calculated 'exact_theta'  

def line(x, theta): 
    return theta[0] + theta[1]*x


# In[24]:


# 5) create new inputs  =>  we create evolution_points  [new i/p = evolution_points]  

evolution_points = np.linspace(0, P[-1], 100).reshape(-1,1)  # create new inputs


# In[25]:


# 6) apply new inputs to our model (line fun) and store new output  =>  we store with name 'y'  [new o/p = y]

y = line(evolution_points, exact_theta)        # new output


# In[26]:


# 7) plot given i/p and o/p 
# 8) in same graph also plot generated new i/p and o/p 
# 9) Check how our new generated data using model, fits the given training data

plt.figure()
plt.plot(P, V, 'kx', markersize=2)                # training data (given data)
plt.plot(evolution_points, y, 'r', label='LR')    # new data (LR model)
plt.xlabel('Pressure [kPa]')
plt.ylabel('Specific Volume [$ cm^{3} g^{-1} $]')
plt.title('Saturated Liquid', size=15)
plt.legend()
plt.show()


# In[ ]:





# In[ ]:




