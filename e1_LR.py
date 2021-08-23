#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as  np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv("DATA_superheated_vapor.csv")  # read and import csv
data.head()


# In[3]:


V_data = data.loc[ data["Property"] == "V" ]
U_data = data.loc[ data["Property"] == "U" ]
H_data = data.loc[ data["Property"] == "H" ]
S_data = data.loc[ data["Property"] == "S" ]

V_data.head()


# In[4]:


# plot saturated liquid

plt.figure( figsize=(13,7) )

plt.subplot(221)
plt.plot(V_data["Pressure"], V_data["Liq_Sat"], 'kx', markersize=3)
plt.xlabel('Pressure [kPa]')
plt.ylabel('Specific volume [$cm^{3} g^{-1}$]')

plt.subplot(222)
plt.plot(U_data["Pressure"], U_data["Liq_Sat"], 'kx', markersize=3)
plt.xlabel('Pressure [kPa]')
plt.ylabel('Specific internal energy [$kJ kg^{-1}$]')

plt.subplot(223)
plt.plot(H_data["Pressure"], H_data["Liq_Sat"], 'kx', markersize=3)
plt.xlabel('Pressure [kPa]')
plt.ylabel('Specific enthalpy [$kJ kg^{-1}$]')

plt.subplot(224)
plt.plot(S_data["Pressure"], S_data["Liq_Sat"], 'kx', markersize=3)
plt.xlabel('Pressure [kPa]')
plt.ylabel('Specific entropy [$kJ kg^{-1} K^{-1}$]')

plt.suptitle('Saturated liquid', size=15)
plt.show()


# In[5]:


# plot saturated vapour

plt.figure( figsize=(13,7) )

plt.subplot(221)
plt.plot(V_data["Pressure"], V_data["Vap_Sat"], 'bx', markersize=3)
plt.xlabel('Pressure [kPA]')
plt.ylabel('Specific volume [$cm^{3} g^{-1}$]')

plt.subplot(222)
plt.plot(U_data["Pressure"], U_data["Vap_Sat"], 'bx', markersize=3)
plt.xlabel('Pressure [kPA]')
plt.ylabel('Specific internal energy [$kJ kg^{-1}$]')

plt.subplot(223)
plt.plot(H_data["Pressure"], H_data["Vap_Sat"], 'bx', markersize=3)
plt.xlabel('Pressure [kPA]')
plt.ylabel('Specific enthalpy [$kJ kg^{-1}$]')

plt.subplot(224)
plt.plot(S_data["Pressure"], S_data["Vap_Sat"], 'bx', markersize=3)
plt.xlabel('Pressure [kPA]')
plt.ylabel('Specific entropy [$kJ kg^{-1} K^{-1}$]')

plt.suptitle('Saturated liquid', size=15)
plt.show()


# In[6]:


plt.figure()
plt.plot(V_data['Pressure'], V_data['Liq_Sat'], 'kx', markersize=3)
plt.axvline(300, linestyle='--', color='r')
plt.axvline(1500, linestyle='--', color='r')
plt.xlabel('Pressure [kPa]')
plt.ylabel('Specific volume [$cm^{3} g^{-1}$]')
plt.show()


# In[7]:


First_P = V_data['Pressure'].loc[ V_data['Pressure'] < 300 ]
First_V = V_data['Liq_Sat'].loc[ V_data['Pressure'] < 300 ]

Second_P = V_data['Pressure'].loc[ (300 <= V_data['Pressure']) & (V_data['Pressure'] < 1500) ]
Second_V = V_data['Liq_Sat'].loc[ (300 <= V_data['Pressure']) & (V_data['Pressure'] < 1500) ]

Third_P = V_data['Pressure'].loc[ 1500 <= V_data['Pressure'] ]
Third_V = V_data['Liq_Sat'].loc[ 1500 <= V_data['Pressure'] ]


# In[8]:


# fun 1 = LR_cost_function

def LR_cost_function(X, y, theta):

    m = X.shape[0]
    h = X.dot(theta)
    y = np.array(y)
    
    J = 1/(2*m) * (h-y.reshape(-1,1)).T.dot(h-y.reshape(-1,1))
    
    grad_J = 1/m * (h-y.reshape(-1,1)).T.dot(X)
    
    return J, grad_J


# fun 2 = line

def line(x, theta):
    x = np.array(x)
    return x*theta[1] + theta[0]


# In[9]:


First_X = np.c_[np.ones((First_P.shape[0], 1)), First_P]
Second_X = np.c_[np.ones((Second_P.shape[0], 1)), Second_P]
Third_X = np.c_[np.ones((Third_P.shape[0], 1)), Third_P]

alpha = 1e-5
theta_i = 2 * np.random.rand(2, 1)
theta_all = []
J_all = []

N = 3000000
for i in range(1,N):
    J_i, grad_J_i = LR_cost_function(First_X, First_V, theta_i)
    theta_i = theta_i - alpha * grad_J_i.T
    J_all.append(J_i)
    theta_all.append(theta_i.T)
    
[J_fin, grad_J_fin] = LR_cost_function(First_X, First_V, theta_i)
J_all.append(J_fin)

print('\n Solution with GD: intercept = ', theta_i[0], ' slope = ', theta_i[1])

J_all = np.array(J_all)
J_all = J_all.reshape(-1,)

plt.figure()
plt.plot(np.linspace(1,N,N), J_all, 'r')
plt.xlabel('Iteration')
plt.ylabel('Cost Function')
plt.title('First Split')
plt.show()


# In[ ]:





# In[10]:


# Using normal function

First_X = np.c_[np.ones((First_P.shape[0], 1)), First_P]
Second_X = np.c_[np.ones((Second_P.shape[0], 1)), Second_P]
Third_X = np.c_[np.ones((Third_P.shape[0], 1)), Third_P]

exact_theta_1 = np.linalg.inv(First_X.T.dot(First_X)).dot(First_X.T).dot(First_V)
exact_theta_2 = np.linalg.inv(Second_X.T.dot(Second_X)).dot(Second_X.T).dot(Second_V)
exact_theta_3 = np.linalg.inv(Third_X.T.dot(Third_X)).dot(Third_X.T).dot(Third_V)

print('\n Exact solution: intercept = ', exact_theta_1[0], ' slope = ', exact_theta_1[1])


# In[ ]:





# In[11]:


# plot LR from Normal Equation 

plt.figure()

plt.xlabel('Pressure [kPa]')
plt.ylabel('Specific volume [$cm^{-3} g^{-1}$]')
plt.title('Using raw numpy implementation')

plt.plot(First_P, First_V, 'rx', markersize=2, label='First split data')     # training data
plt.plot(Second_P, Second_V, 'bx', markersize=2, label='Second split data')  # training data
plt.plot(Third_P, Third_V, 'kx', markersize=2, label='First split data')     # training data

d1 = np.linspace(0,1000, 100)       # temp length 
d2 = np.linspace(300,3000, 100)     # temp length 
d3 = np.linspace(2000,11500, 100)   # temp length 
plt.plot(d1, line(d1,exact_theta_1), 'r', linewidth=1, label='First LR')   # using normal equation
plt.plot(d2, line(d2,exact_theta_2), 'b', linewidth=1, label='Second LR')  # using normal equation
plt.plot(d3, line(d3,exact_theta_3), 'k', linewidth=1, label='Third LR')   # using normal equation

plt.legend()
plt.show()


# In[ ]:




