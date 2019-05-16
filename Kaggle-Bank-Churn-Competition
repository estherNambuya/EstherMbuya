#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import pandas


# In[2]:


url = 'Churn_Modelling.csv'
names = ['Age', 'CreditScore', 'Tenure', 'NumOfProducts', 'HasCrCard','IsActiveMember','EstimatedSalary','Exited']
dataset = pandas.read_csv(url, names=names)


# In[3]:


print(dataset.shape)


# In[4]:


dataset.head()


# In[5]:


dataset.describe()


# In[6]:


dataset.groupby('Exited').size()


# In[7]:


dataset['Exited'].value_counts()


# In[8]:


import numpy as np
from plotly import __version__


# In[9]:


import cufflinks as cf
import pandas as pd


# In[10]:


from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot


# In[11]:


init_notebook_mode(connected=True)


# In[12]:


cf.go_offline()


# In[13]:


dataset.iplot


# In[27]:


dataset.iplot(kind='box')


# In[23]:


dataset.iplot(kind='hist')


# In[44]:


dataset.iplot(kind='scatter', x='Age',y='Exited',mode='markers',size=10)


# In[45]:


dataset.iplot(kind='scatter', x='NumOfProducts',y='Exited',mode='markers',size=10)


# In[46]:


dataset.iplot(kind='bo')


# In[ ]:





# In[ ]:





# In[ ]:




