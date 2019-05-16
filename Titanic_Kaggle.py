#!/usr/bin/env python
# coding: utf-8

# In[56]:


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


# In[57]:


url = "Titanic/train.csv"
names = ['Passenger', 'Pclass', 'Name', 'Sex', 'Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked','Survived']
dataset = pandas.read_csv(url, names=names)


# In[58]:


url2 = "Titanic/test.csv"
names = ['Passenger', 'Pclass', 'Name', 'Sex', 'Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked','Survived']
dataset = pandas.read_csv(url, names=names)


# In[59]:


print(dataset.shape)


# In[60]:


dataset.head()


# In[61]:


dataset.describe()


# In[62]:


dataset.groupby('Survived').size()


# In[63]:


dataset['Survived'].value_counts()


# In[100]:


dataset = dataset[~dataset['Survived'].isnull()]


# In[101]:


dataset[['Survived']] = dataset[['Survived']].astype(float)


# In[102]:


dataset[['Survived']] = dataset[['Survived']].astype(int)


# In[103]:


sns.distplot(dataset['Survived'])


# In[81]:


dataset = dataset[~dataset['Age'].isnull()]


# In[82]:


dataset[['Age']] = dataset[['Age']].astype(int)


# In[83]:


sns.distplot(dataset['Age'])


# In[107]:


dataset = dataset[~dataset['Fare'].isnull()]


# In[108]:


dataset[['Fare']] = dataset[['Fare']].astype(int)


# In[109]:


sns.distplot(dataset['Fare'])


# In[110]:


dataset = dataset[~dataset['Ticket'].isnull()]


# In[ ]:


dataset[['Ticket']] = dataset[['Ticket']].astype(int)


# In[111]:


sns.distplot(dataset['Ticket'])


# In[112]:


dataset = dataset[~dataset['Pclass'].isnull()]


# In[ ]:


dataset[['Pclass']] = dataset[['Pclass']].astype(int)


# In[113]:


sns.distplot(dataset['Pclass'])


# In[ ]:





# In[86]:


dataset['Pclass'] = dataset['Pclass'].convert_objects(convert_numeric=True)


# In[87]:


dataset['Age'] = dataset['Age'].convert_objects(convert_numeric=True)


# In[88]:


dataset['Parch'] = dataset['Parch'].convert_objects(convert_numeric=True)


# In[89]:


dataset['Cabin'] = dataset['Cabin'].convert_objects(convert_numeric=True)


# In[90]:


dataset['Embarked'] = dataset['Embarked'].convert_objects(convert_numeric=True)


# In[91]:


dataset['Survived'] = dataset['Survived'].convert_objects(convert_numeric=True)


# In[92]:


dataset['Fare'] = dataset['Fare'].convert_objects(convert_numeric=True)


# In[93]:


dataset['Ticket'] = dataset['Ticket'].convert_objects(convert_numeric=True)


# In[94]:


dataset['SibSp'] = dataset['SibSp'].convert_objects(convert_numeric=True)


# In[95]:


scatter_matrix(dataset)
plt.show()


# In[96]:


import cufflinks as cf
import seaborn as sns 


# In[ ]:





# In[ ]:





# In[97]:


from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot


# In[98]:


init_notebook_mode(connected=True)


# In[99]:


cf.go_offline()


# In[27]:


dataset.iplot


# In[28]:


dataset.iplot(kind='bar')


# In[29]:


dataset = dataset[~dataset['Age'].isnull()]


# In[30]:


dataset[['Age']] = dataset[['Age']].astype(int)


# In[31]:


dataset = dataset[~dataset['Survived'].isnull()]


# In[32]:


dataset[['Survived']] = dataset[['Survived']].astype(int)


# In[33]:


dataset.iplot(kind='scatter')


# In[34]:


datasett = dataset[~dataset['Cabin'].isnull()]
datasett['Cabin'] = dataset['Cabin'].convert_objects(convert_numeric=True)
datasett = dataset[~dataset['SibSp'].isnull()]
datasett['SibSp'] = dataset['SibSp'].convert_objects(convert_numeric=True)
datasett = dataset[~dataset['Parch'].isnull()]
datasett['Parch'] = dataset['Parch'].convert_objects(convert_numeric=True)
datasett = dataset[~dataset['Ticket'].isnull()]
datasett['Ticket'] = dataset['Ticket'].convert_objects(convert_numeric=True)
datasett = dataset[~dataset['Fare'].isnull()]
datasett['Fare'] = dataset['Fare'].convert_objects(convert_numeric=True)
datasett = dataset[~dataset['Embarked'].isnull()]
datasett['Embarked'] = dataset['Embarked'].convert_objects(convert_numeric=True)
array = datasett.values
X = array[:,5:11]
Y = array[:,11]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


# In[35]:



X


# In[36]:


Y


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




