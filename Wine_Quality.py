#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import numpy as np
import matplotlib.pyplot as plt  
import pandas as pd
import seaborn as sns

from warnings import filterwarnings
filterwarnings(action='ignore')


# # Loading Dataset

# In[2]:


wine = pd.read_csv("winequality-red.csv")
print("Successfully Imported Data!")
wine.head()


# In[3]:


print(wine.shape)


# # Description

# In[4]:


wine.describe(include='all')


# # Finding Null Values

# In[5]:


print(wine.isna().sum())


# In[6]:


wine.corr()


# In[7]:


wine.groupby('quality').mean()


# # Data Analysis

# ## Countplot:

# In[8]:


sns.countplot(wine['quality'])
plt.show()


# In[9]:


sns.countplot(wine['pH'])
plt.show()


# In[10]:


sns.countplot(wine['alcohol'])
plt.show()


# In[11]:


sns.countplot(wine['fixed acidity'])
plt.show()


# In[12]:


sns.countplot(wine['volatile acidity'])
plt.show()


# In[13]:


sns.countplot(wine['citric acid'])
plt.show()


# In[14]:


sns.countplot(wine['density'])
plt.show()


# ## KDE plot:

# In[15]:


sns.kdeplot(wine.query('quality > 2').quality)


# ## Distplot:

# In[16]:


sns.distplot(wine['alcohol'])


# In[17]:


wine.plot(kind ='box',subplots = True, layout =(4,4),sharex = False)


# In[18]:


wine.plot(kind ='density',subplots = True, layout =(4,4),sharex = False)


# ## Histogram

# In[19]:


wine.hist(figsize=(10,10),bins=50)
plt.show()


# # Heatmap for expressing correlation

# In[20]:


corr = wine.corr()
sns.heatmap(corr,annot=True)


# ## Pair Plot:

# In[21]:


sns.pairplot(wine)


# ## Violinplot:

# In[22]:


sns.violinplot(x='quality', y='alcohol', data=wine)


# # Feature Selection

# In[23]:


# Create Classification version of target variable
wine['goodquality'] = [1 if x >= 7 else 0 for x in wine['quality']]# Separate feature variables and target variable
X = wine.drop(['quality','goodquality'], axis = 1)
Y = wine['goodquality']


# In[24]:


# See proportion of good vs bad wines
wine['goodquality'].value_counts()


# In[25]:


X


# In[26]:


print(Y)


# # Feature Importance

# In[36]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

from sklearn.ensemble import ExtraTreesClassifier
classifiern = ExtraTreesClassifier()
classifiern.fit(X,Y)
score = classifiern.feature_importances_
print(score)


# # Splitting Dataset

# In[37]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=7)


# # LogisticRegression:

# In[38]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix
print("Accuracy Score:",accuracy_score(Y_test,Y_pred))


# In[39]:


confusion_mat = confusion_matrix(Y_test,Y_pred)
print(confusion_mat)


# # Using KNN:

# In[40]:


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train,Y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(Y_test,y_pred))


# # Using SVC:

# In[41]:


from sklearn.svm import SVC
model = SVC()
model.fit(X_train,Y_train)
pred_y = model.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(Y_test,pred_y))


# # Using Decision Tree:

# In[42]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion='entropy',random_state=7)
model.fit(X_train,Y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(Y_test,y_pred))


# # Using GaussianNB:

# In[43]:


from sklearn.naive_bayes import GaussianNB
model3 = GaussianNB()
model3.fit(X_train,Y_train)
y_pred3 = model3.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(Y_test,y_pred3))


# # Using Random Forest:

# In[44]:


from sklearn.ensemble import RandomForestClassifier
model2 = RandomForestClassifier(random_state=1)
model2.fit(X_train, Y_train)
y_pred2 = model2.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(Y_test,y_pred2))


# # Using Xgboost:

# In[45]:


import xgboost as xgb
model5 = xgb.XGBClassifier(random_state=1)
model5.fit(X_train, Y_train)
y_pred5 = model5.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(Y_test,y_pred5))


# In[46]:


results = pd.DataFrame({
    'Model': ['Logistic Regression','KNN', 'SVC','Decision Tree' ,'GaussianNB','Random Forest','Xgboost'],
    'Score': [0.870,0.872,0.868,0.864,0.833,0.893,0.879]})

result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df


# In[47]:


#Hence I will use Random Forest algorithms for training my model.


# In[ ]:




