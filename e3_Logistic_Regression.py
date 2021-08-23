#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt

# sklearn
from sklearn.linear_model  import LogisticRegression 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict

from sklearn.dummy import DummyClassifier

from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import precision_recall_curve, roc_curve 


# ## 1. Generate random data to train the model

# In[3]:


random_seed_value = 42;
np.random.seed(random_seed_value)

m1 = 400
m2 = 1200
m3 = 100

x_1 = 0.021 * np.random.rand(m1,1)
x_2 = 0.0115 * np.random.rand(m2,1) + 0.0195
x_3 = 0.008 * np.random.rand(m3,1) + 0.0295
X = np.vstack(( x_1, x_2, x_3 ))
y = np.vstack(( np.zeros(x_1.shape), np.ones(x_2.shape), np.zeros(x_3.shape) ))


# ## 2. Split the data into training and test dataset

# In[4]:


# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed_value)

# plot training data
plt.figure( figsize=(15,4) )
plt.plot(X_train[y_train==0], y_train[y_train==0], "s", color="blue")
plt.plot(X_train[y_train==1], y_train[y_train==1], "s", color="darkorange")
plt.axis([0, 0.04, -0.02, 1.02])
plt.yticks(ticks=np.array([0,1]), labels=["Waste", "OK"])
plt.title("Training data", size=15)
plt.show()

# plot all data
plt.figure( figsize=(15,4) )

plt.plot(x_1, np.zeros(x_1.shape), "s", color="blue")
plt.plot(x_2, np.ones(x_2.shape), "s", color="darkorange")
plt.plot(x_3, np.zeros(x_3.shape), "s", color="blue")

plt.axis([0, 0.04, -0.02, 1.02])
plt.yticks(ticks=np.array([0, 1]), labels=["Waste", "OK"])
plt.title("All data", size=15)
plt.show()


# ## 3. Dummy classifier

# In[5]:


DC = DummyClassifier(strategy="constant", constant=1)   # initiate DC

DC.fit(X_train, y_train)                                # train
DC_pred_y_train = DC.predict(X_train)                   # predict

tn, fp, fn, tp = confusion_matrix(y_train, DC_pred_y_train).ravel() 

print("--- Dummy Classifier --- \n")
print("Confusion Matrix")
print(np.array([ [tp, fp], [fn, tn] ]))
print(" ")
print("True  Negative : {0:0.0f}".format(tn))
print("False Positive : {0:0.0f}".format(fp))
print("False Negative : {0:0.0f}".format(fn))
print("True  Positive : {0:0.0f}".format(tp))
print("Precision score: {0:0.2f}".format( precision_score(y_train, DC_pred_y_train) ))
print("Recall score   : {0:0.2f}".format(    recall_score(y_train, DC_pred_y_train) ))
print("f1 score       : {0:0.2f}".format(        f1_score(y_train, DC_pred_y_train) ))


# ## 4. Logistic Regression (Degree - 1)

# In[6]:


ss1 = StandardScaler()
logi_reg1 = LogisticRegression(solver="liblinear", random_state=random_seed_value)

only_preprocessing1 = Pipeline([ ("std_scaler", ss1) ])
logi_reg_d1 = Pipeline([ ("std_scaler", ss1), ("logi_reg1", logi_reg1) ])

# train model
only_preprocessing1.fit(X_train)
logi_reg_d1.fit(X_train, y_train.ravel())    # logiReg d1 model

# new dataset only for visulation of function
X_new = np.linspace(-0.002,0.037, 100).reshape(-1,1)

# scaling
X_train_ss = only_preprocessing1.transform(X_train)
X_new_ss   = only_preprocessing1.transform(X_new)

X_new_ss_ones = np.hstack(( np.ones((X_new_ss.shape[0], 1)), X_new_ss ))
theta         = np.vstack(( logi_reg1.intercept_, logi_reg1.coef_ ))
y_new    = 1 / (1 + np.exp( -X_new_ss_ones.dot(theta) ))

# plot new data to visulize function
plt.figure( figsize=(15,4) )

plt.plot(X_train_ss[y_train==0], y_train[y_train==0], "s", color="blue")
plt.plot(X_train_ss[y_train==1], y_train[y_train==1], "s", color="darkorange")
plt.plot(X_new_ss, y_new, "g", lw=3)
plt.plot([-3, 2], [0.5, 0.5], "k--", lw=1)

plt.axis([-3, 2, -0.02, 1.02])
plt.yticks(ticks=np.array([0,1]), labels=["Waste", "OK"])
plt.show()


# #### Cross Validation

# In[57]:


# CV  =>  cross_val_predict(model, X, y, num_of_CV_fold)

y_train_cv = cross_val_predict(logi_reg_d1, X_train, y_train.ravel(), cv=5)

tn, fp, fn, tp = confusion_matrix(y_train, y_train_cv).ravel() 

print("--- Logistic Regression d1 --- \n")
print("Confusion Matrix")
print(np.array([ [tp, fp], [fn, tn] ]))
print(" ")
print("True  Negative : {0:0.0f}".format(tn))
print("False Positive : {0:0.0f}".format(fp))
print("False Negative : {0:0.0f}".format(fn))
print("True  Positive : {0:0.0f}".format(tp))
print("Precision score: {0:0.2f}".format( precision_score(y_train, y_train_cv) ))
print("Recall score   : {0:0.2f}".format(    recall_score(y_train, y_train_cv) ))
print("f1 score       : {0:0.2f}".format(        f1_score(y_train, y_train_cv) ))


# #### Precision - Recall curve

# In[58]:


y_test_pred = logi_reg_d1.predict_proba(X_test)[:,1]

precision, recall, threshold = precision_recall_curve(y_test, y_test_pred)

plt.figure( figsize=(5,5) )
plt.plot(recall[:-1], precision[:-1], marker=".")
plt.xlabel("recall")
plt.ylabel("precision")
plt.show()


# #### ROC curve and ROC_AUC score

# In[59]:


fp, tp, threshold_roc1 = roc_curve(y_test, y_test_pred)
auc                    = roc_auc_score(y_test, y_test_pred)

plt.figure( figsize=(5,5) )
plt.plot(fp, tp, marker=".")
plt.text(0.7, 0.1, "AUC : %0.4f" %auc)
plt.xlabel("False Positive")
plt.ylabel("True Positive")
plt.show()


# ## 5. Logistic Regression (Degree - 2)

# In[60]:


poly_features = PolynomialFeatures(degree=2, include_bias=False)
ss2 = StandardScaler()

logi_reg2 = LogisticRegression(solver="lbfgs", random_state=random_seed_value, penalty="none")

only_preprocessing2 = Pipeline([ ("poly_features", poly_features), ("std_scaler2", ss2)])
logi_reg_d2         = Pipeline([ ("poly_features", poly_features), ("std_scaler2", ss2), ("logi_reg2", logi_reg2)])

# train model
only_preprocessing2.fit(X_train)
logi_reg_d2.fit(X_train, y_train.ravel())    # logiReg d1 model

# scaling and creating polynomial features using training data

X_train_ss = only_preprocessing2.transform(X_train)  # scale & create features
X_train_f1 = X_train_ss[:,0].reshape(-1,1)           # feature 1
X_train_f2 = X_train_ss[:,1].reshape(-1,1)           # feature 2

# plotting
left_right   = np.array([-2.5, 2])
boundary_reg = -(logi_reg2.coef_[0][0] * left_right + logi_reg2.intercept_[0] - 0.5) / logi_reg2.coef_[0][1]

plt.figure( figsize=(5,4) )

plt.plot(X_train_f1[y_train==0], X_train_f2[y_train==0], "s", color="blue")
plt.plot(X_train_f1[y_train==1], X_train_f2[y_train==1], "s", color="darkorange")
plt.plot(left_right, boundary_reg, "g", lw=3)

plt.show()


# #### Cross Validation

# In[61]:


# CV  =>  cross_val_predict(model, X, y, num_of_CV_fold)

y_train_cv2 = cross_val_predict(logi_reg_d2, X_train, y_train.ravel(), cv=5)

tn, fp, fn, tp = confusion_matrix(y_train, y_train_cv2).ravel()

print("--- Logistic Regression d1 --- \n")
print("Confusion Matrix")
print(np.array([ [tp, fp], [fn, tn] ]))
print(" ")
print("True  Negative : {0:0.0f}".format(tn))
print("False Positive : {0:0.0f}".format(fp))
print("False Negative : {0:0.0f}".format(fn))
print("True  Positive : {0:0.0f}".format(tp))
print("Precision score: {0:0.2f}".format( precision_score(y_train, y_train_cv2) ))
print("Recall score   : {0:0.2f}".format(    recall_score(y_train, y_train_cv2) ))
print("f1 score       : {0:0.2f}".format(        f1_score(y_train, y_train_cv2) ))


# #### Precision-Recall curve

# In[62]:


y_test_pred2 = logi_reg_d2.predict_proba(X_test)[:,1]

precision, recall, threshold = precision_recall_curve(y_test, y_test_pred2)

plt.figure( figsize=(5,5) )
plt.plot(recall[:-1], precision[:-1], marker=".")
plt.xlabel("recall")
plt.ylabel("precision")
plt.show()


# #### ROC curve and ROC_AUC score

# In[63]:


fp, tp, threshold_roc1 = roc_curve(y_test, y_test_pred2)
auc                    = roc_auc_score(y_test, y_test_pred2)

plt.figure( figsize=(5,5) )
plt.plot(fp, tp, marker=".")
plt.text(0.7, 0.1, "AUC : %0.4f" %auc)
plt.xlabel("False Positive")
plt.ylabel("True Positive")
plt.show()


# In[ ]:




