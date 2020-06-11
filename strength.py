#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[60]:


dataset = pd.read_csv(r"EmployeeAttrition.csv")


# In[61]:


dataset.isnull().any()


# In[62]:


dataset


# In[64]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dataset['Education'] = le.fit_transform(dataset['Education'])
dataset['JobInvolvement'] = le.fit_transform(dataset['JobInvolvement'])
dataset['PerformanceRating'] = le.fit_transform(dataset['PerformanceRating'])
dataset['Attrition(Yes/No)'] = le.fit_transform(dataset['Attrition(Yes/No)'])
#dataset[''] = le.fit_transform(dataset[''])


# In[65]:


dataset


# In[66]:


x = dataset.iloc[:,:13].values
y = dataset.iloc[:,13].values


# In[67]:


x


# In[68]:


y


# In[69]:


from sklearn.preprocessing import OneHotEncoder
oh = OneHotEncoder()
z = oh.fit_transform(x).toarray()


# In[70]:


x


# In[71]:


X = x[:,1:]


# In[72]:


X


# In[73]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)


# In[74]:


X_train


# In[75]:


y_train


# In[76]:


X_test


# In[77]:


y_test


# In[78]:


X_train.shape


# In[79]:


y_train.shape


# In[80]:


from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression()
regressor.fit(X_train,y_train)


# In[81]:


y_pred = regressor.predict(X_test)


# In[82]:


y_pred


# In[93]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[94]:


from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


# In[95]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion = 'entropy',random_state = 0)
model.fit(X_train,y_train)


# In[96]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)


# In[97]:


cm


# In[100]:


import sklearn.metrics as metrics
fpr,tpr,threshold = metrics.roc_curve(y_test,y_pred)
roc_auc = metrics.auc(fpr,tpr)


# In[101]:


roc_auc


# In[92]:


import matplotlib.pyplot as plt
plt.title("roc")
plt.plot(fpr,tpr,'b',label = 'auc = %0.2f'%roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('tpr')
plt.xlabel('fpr')


# In[ ]:





# In[ ]:




