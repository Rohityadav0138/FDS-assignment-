#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

#Import Cancer data from the Sklearn library
# Dataset can also be found here (http://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+%28diagnostic%29)
# NAme-Rohit yadav 2018A7PS0138U
#name-Naman Gupta 2018A7PS0198U

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()


# In[2]:


df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'], ['target']))

df_cancer.head()


# In[3]:


df_cancer.shape


# In[4]:


df_cancer.columns


# In[5]:


sns.pairplot(df_cancer, vars = ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
       'mean smoothness'] )


# In[6]:


sns.pairplot(df_cancer, hue = 'target', vars = ['mean radius', 'mean texture', 'mean perimeter','mean area','mean smoothness'] )


# In[7]:


df_cancer['target'].value_counts()


# In[8]:


sns.countplot(df_cancer['target'], label = "Count") 


# In[9]:


plt.figure(figsize=(20,12)) 
sns.heatmap(df_cancer.corr(), annot=True) 


# In[10]:


X = df_cancer.drop(['target'], axis = 1) # We drop our "target" feature and use all the remaining features in our dataframe to train the model.
X.head()


# In[11]:


y = df_cancer['target']
y.head()


# In[12]:


from sklearn.model_selection import train_test_split


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 20)


# In[14]:


print ('The size of our training "X" (input features) is', X_train.shape)
print ('\n')
print ('The size of our testing "X" (input features) is', X_test.shape)
print ('\n')
print ('The size of our training "y" (output feature) is', y_train.shape)
print ('\n')
print ('The size of our testing "y" (output features) is', y_test.shape)


# In[15]:


from sklearn.svm import SVC


# In[18]:


svc_model = SVC()


# In[20]:


svc_model.fit(X_train, y_train)


# In[21]:


y_predict = svc_model.predict(X_test)


# In[22]:


# Import metric libraries

from sklearn.metrics import classification_report, confusion_matrix


# In[23]:


cm = np.array(confusion_matrix(y_test, y_predict, labels=[1,0]))
confusion = pd.DataFrame(cm, index=['is_cancer', 'is_healthy'],
                         columns=['predicted_cancer','predicted_healthy'])
confusion


# In[24]:


sns.heatmap(confusion, annot=True)


# In[25]:


print(classification_report(y_test, y_predict))


# In[26]:


# these steps are only for improving our data by normlization
# Name1- Rohit yadav 2018A7PS0138U
#Name2- Naman gupta 2018A7PS0198U

# FIrst process is normalization

X_train_min = X_train.min()
X_train_min


# In[27]:


X_train_max = X_train.max()
X_train_max


# In[28]:


X_train_range = (X_train_max- X_train_min)
X_train_range


# In[29]:


X_train_scaled = (X_train - X_train_min)/(X_train_range)
X_train_scaled.head()


# In[30]:


# normalize training data set

X_test_min = X_test.min()
X_test_range = (X_test - X_test_min).max()
X_test_scaled = (X_test - X_test_min)/X_test_range


# In[31]:


svc_model = SVC()
svc_model.fit(X_train_scaled, y_train)


# In[32]:


y_predict = svc_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_predict)


# In[33]:


# SVM with normalized data

cm = np.array(confusion_matrix(y_test, y_predict, labels=[1,0]))
confusion = pd.DataFrame(cm, index=['is_cancer', 'is_healthy'],
                         columns=['predicted_cancer','predicted_healthy'])
confusion


# In[34]:


sns.heatmap(confusion,annot=True,fmt="d")


# In[35]:


print(classification_report(y_test,y_predict))


# In[36]:


# TO improve the performance we use GRIDSEARCH method

param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']} 


# In[37]:


from sklearn.model_selection import GridSearchCV


# In[38]:


grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=4)


# In[39]:


grid.fit(X_train_scaled,y_train)


# In[40]:


#Let's print out the "grid" with the best parameter

print (grid.best_params_)
print ('\n')
print (grid.best_estimator_)


# In[41]:


grid_predictions = grid.predict(X_test_scaled)


# In[42]:


cm = np.array(confusion_matrix(y_test, grid_predictions, labels=[1,0]))
confusion = pd.DataFrame(cm, index=['is_cancer', 'is_healthy'],
                         columns=['predicted_cancer','predicted_healthy'])
confusion


# In[43]:


sns.heatmap(confusion, annot=True)


# In[44]:


print(classification_report(y_test,grid_predictions))


# In[ ]:


# As we can see, our best model is SVM with Normalized data, followed by our Gridsearch model

