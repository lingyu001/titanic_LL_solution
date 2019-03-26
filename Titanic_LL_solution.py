#!/usr/bin/env python
# coding: utf-8

# # Titanic LL solution

# #### Attempt the solution of Titanic Data from https://www.kaggle.com/c/titanic using Logistic Regression, Regression Tree and SVM.

# ## Import libs

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ## Explore the data

# In[2]:


tt_train = pd.read_csv('train.csv')
tt_test = pd.read_csv('test.csv')


# In[3]:


# tt = tt_train.append(tt_test, ignore_index=True)
tt = pd.concat([tt_train,tt_test])
tt = tt.set_index('PassengerId')


# In[4]:


tt.info()


# In[38]:


tt_train.info()


# In[5]:


tt.head()


# In[ ]:


tt_train.describe()


# In[20]:


#tt_train.info()
tt.info()


# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=tt_train)


# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=tt_train)


# In[76]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Embarked',data=tt_train)


# In[ ]:


plt.figure(figsize=(10,6))
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='SibSp',data=tt_train,)
plt.legend(loc='upper right')


# In[93]:


plt.figure(figsize=(10,6))
#sns.distplot(tt_train['Age'].dropna(),hue='Survived',kde=False,color='blue',bins=30)
plt.hist(tt_train['Age'][tt_train['Survived']==0],bins=25, alpha=0.5, label='non-survived', color='red')
plt.hist(tt_train['Age'][tt_train['Survived']==1],bins=25, alpha=0.5, label='survived', color='blue')


# In[34]:


plt.figure(figsize=(10,6))
sns.pairplot(tt_train['Fare'].dropna(),kde=False,color='blue',bins=30)


# In[35]:


tt_train['Parch'].value_counts()


# ## Check Missing data

# In[6]:


#sns.heatmap(tt_train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.heatmap(tt_test.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[23]:


# Check Age
plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=tt_train,palette='winter')


# In[221]:


# Check Cabin
#cabin_index = tt_train['Cabin'].astype(str).str[0]
cabin_index = tt_train['Cabin'].astype(str).str[0]


# In[223]:


sns.set_style('whitegrid')
# sns.countplot(x='Survived',hue=cabin_index,data=tt_train)
sns.countplot(x='Survived',hue=cabin_index,data=tt)


# In[222]:


#tt_train.groupby(cabin_index).mean()
tt.groupby(cabin_index).mean()


# ## Extract Title from Name

# In[7]:


# tt_train['Title'] = tt_train['Name'].str.split(',').apply(lambda x: x[1])
# title = tt_train['Name'].str.split(',').apply(lambda x: x[1])
title = tt['Name'].str.split(',').apply(lambda x: x[1])


# In[8]:


title = title.str.split('.').apply(lambda x: x[0])


# In[9]:


title.value_counts()


# In[12]:


title_cat = {
    ' Mr': 'Mr',
    ' Miss': 'Miss',
    ' Mrs': 'Mrs',
    ' Ms': 'Miss',
    ' Master': 'Master',
    ' Dr': 'Officer',
    ' Rev': 'Officer',
    ' Col': 'Officer',
    ' Major': 'Officer',
    ' Mlle': 'Miss',
    ' Don': 'Noble',
    ' Dona':'Noble',
    ' Sir': 'Noble',
    ' the Countess': 'Noble',
    ' Mme': 'Mrs',
     ' Capt': 'Officer',
     ' Jonkheer': 'Noble',
     ' Lady': 'Noble'
       }


# In[13]:


title_update = title.map(title_cat)


# In[14]:


title_update.isna().sum()


# In[250]:


tt[title_update.isna()]


# In[46]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue=title_update,data=tt_train)


# In[80]:


title_update.value_counts()


# ## Age Imputation

# In[254]:


# tt_train['Age'].groupby(tt_train['Pclass']).median()
tt['Age'].groupby(tt['Pclass']).median()


# In[13]:


tt_train['Age'].groupby(tt_train['Pclass']).mean()


# In[14]:


tt_test['Age'].groupby(tt_test['Pclass']).mean()


# In[15]:


#tt_train['Age'].groupby([tt_train['Sex'],tt_train['Pclass'],title_update]).median()
tt['Age'].groupby([tt['Sex'],tt['Pclass'],title_update]).median()


# In[60]:


# Imputation Age by Pclass
# def impute_age(cols):
#     Age = cols[0]
#     Pclass = cols[1]
    
#     if pd.isnull(Age):

#         if Pclass == 1:
#             return tt_train['Age'][tt_train['Pclass']==1].mean()

#         elif Pclass == 2:
#             return tt_train['Age'][tt_train['Pclass']==2].mean()

#         else:
#             return tt_train['Age'][tt_train['Pclass']==3].mean()

#     else:
#         return Age


# In[11]:


# Imputate Age
#tt_train['Age'] = tt_train[['Age','Pclass']].apply(impute_age,axis=1)


# In[252]:


# Imputation by pclass
# pclass_group = tt_train.groupby('Pclass')
pclass_group = tt.groupby('Pclass')


# In[253]:


pclass_group.Age.median()


# In[255]:


# tt_train['Age'] = pclass_group.Age.apply(lambda x: x.fillna(x.median()))
tt['Age'] = pclass_group.Age.apply(lambda x: x.fillna(x.median()))


# In[16]:


#Imputation by sex,pclass and title
# age_group = tt_train.groupby(['Sex','Pclass', title_update]) 
age_group = tt.groupby(['Sex','Pclass', title_update]) 


# In[17]:


# tt_train['Age'] = age_group.Age.apply(lambda x: x.fillna(x.median()))
tt['Age'] = age_group.Age.apply(lambda x: x.fillna(x.median()))


# In[18]:


# tt_train.Age.isna().sum()
tt['Age'].isna().sum()


# In[19]:


tt[tt['Age'].isna()]


# ## Cabin Imputation

# In[20]:


# Imputate Carbin, grab first letter
# cabin_index = tt_train.Cabin.astype(str).str[0]
cabin_index = tt.Cabin.astype(str).str[0]


# In[21]:


cabin_index.value_counts()


# ## Fare Imputation

# In[22]:


# Impute with the median Fare for the same Pclass
tt['Fare'][tt['Fare'].isna()] = tt['Fare'][tt['Pclass']==3].median()


# In[23]:


tt[tt['Fare'].isna()]


# ## Embarked imputation by adjacent ticket number

# In[24]:


tt['Embarked'][tt['Embarked'].isna()] = 'S'


# ## Create dummies for categorical variables

#     1. Sex: convert to male, female, create dummy
#     2. Children: Age < 16
#     2. Family size: create category according to Sibsip and Parch
#     3. Pclass: create dummy
#     4. Embark: create dummy
#     5. Cabin index: create dummy

# In[25]:


#Sex dummy
# sex = pd.get_dummies(tt_train['Sex'],drop_first=True)
sex = pd.get_dummies(tt['Sex'],drop_first=True)


# In[26]:


sex.head()


# In[27]:


# Children dummy
# child = pd.get_dummies(tt_train['Age'] < 16,prefix='child',drop_first=True)
child = pd.get_dummies(tt['Age'] < 16,prefix='child',drop_first=True)


# In[28]:


child.head()


# In[68]:


# Combine Child and Sex variable
sex['male'][child['child_True']==1]= 0


# In[34]:


# Family size
# familysize = 1 + tt_train['Parch'] + tt_train['SibSp']

tt['familysize'] = 1 + tt['Parch'] + tt['SibSp']


# In[37]:


tt['familysize'].head()


# In[38]:


# Pclass dummy
# pclass = pd.get_dummies(tt_train['Pclass'],prefix='pclass',drop_first=True)
pclass = pd.get_dummies(tt['Pclass'],prefix='pclass',drop_first=True)


# In[39]:


pclass.head()


# In[40]:


# Embark dummy
# embark = pd.get_dummies(tt_train['Embarked'],prefix='embark',drop_first=True)
embark = pd.get_dummies(tt['Embarked'],prefix='embark',drop_first=True)


# In[41]:


# Cabin index dummy
cabin = pd.get_dummies(cabin_index,prefix='cabin',drop_first=True)


# In[42]:


cabin.head()


# ### Produce the analytic data

# In[43]:


tt.head()


# In[51]:


# tt_train = pd.concat([tt_train['Survived'],tt_train['Age'],tt_train['Fare'],sex,child,familysize,pclass,embark,cabin],axis=1)
tt = pd.concat([tt['Survived'],tt['Age'],tt['Fare'],sex,child,tt['familysize'],
                tt['Parch'],tt['SibSp'],pclass,embark,cabin],axis=1)


# In[84]:


tt['male']=sex['male'].values


# In[85]:


pd.crosstab(tt['male'],tt['child_True'])


# In[163]:


#tt_train.to_csv('tt_train.csv') 
#tt_train_2 = pd.read_csv('tt_train.csv')


# In[15]:


#tt_train = pd.concat([tt_train,title_dummy],axis=1)


# In[291]:


from sklearn.model_selection import train_test_split


# In[52]:


X = tt_train.drop('Survived',axis=1)
y = tt_train['Survived']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=101)


# In[89]:


X_train = tt[tt['Survived'].notnull()].drop('Survived',axis=1)
y_train = tt[tt['Survived'].notnull()]['Survived']
X_test = tt[tt['Survived'].isna()].drop('Survived',axis=1)
y_test = tt[tt['Survived'].isna()]['Survived']


# In[54]:


X_train.info()


# In[57]:


X_test.info()


# In[59]:


y_train.head()


# ## Logistic Regression

# In[313]:


from sklearn.linear_model import LogisticRegression


# In[314]:


logit_model = LogisticRegression()
logit_model.fit(X_train,y_train)


# In[315]:


logit_pred = logit_model.predict(X_test)


# In[56]:


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print(classification_report(y_test,logit_pred))
print(confusion_matrix(y_test,logit_pred))
print('\n','accuracy')
print(accuracy_score(y_test,logit_pred))


# # SVM 

# In[57]:


from sklearn.svm import SVC


# In[58]:


svm_model=SVC(kernel='rbf',C=1000000,gamma=0.00001)
#'linear','poly','rbf','sigmoid','precomputed'


# In[59]:


svm_model.fit(X_train,y_train)


# In[60]:


svm_pred = svm_model.predict(X_test)


# In[61]:


from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,svm_pred))
print(confusion_matrix(y_test,svm_pred))
print('\n','accuracy')
print(accuracy_score(y_test,svm_pred))


# ### Gridsearch

# In[279]:


from sklearn.model_selection import GridSearchCV


# #### SVM with rbf kernel

# In[280]:


param_rbf = {'C': [100, 1000,10000,100000,1000000], 'gamma': [0.01,0.001,0.0001,0.00001,0.000001], 'kernel': ['rbf']} 
#'linear','poly','rbf','sigmoid','precomputed'


# In[281]:


grid_rbf = GridSearchCV(SVC(),param_rbf,refit=True,verbose=3)


# In[1]:


grid_rbf.fit(X_train,y_train)


# In[285]:


grid_rbf.best_params_


# In[286]:


grid_rbf.best_estimator_


# In[287]:


grid_rbf_pred = grid_rbf.predict(X_test)


# In[288]:


from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,grid_rbf_pred))
print(confusion_matrix(y_test,grid_rbf_pred))
print('\n','accuracy')
print(accuracy_score(y_test,grid_rbf_pred))


# #### SVM with poly kernel

# In[18]:


#param_poly = {'C': [100],'gamma': [1,0.1],
             # 'degree': [1],'kernel': ['poly']} 
#param_poly = {'C': [0.1,1, 10, 100, 1000,10000,100000],'gamma': [100,10,1,0.1,0.01,0.001,0.0001,0.00001,0.000001],
     #         'degree': [0,1,2,3,4,5,6],'kernel': ['poly']} 
param_poly = {'C': [0.1,1, 10, 100, 1000,10000,100000],'gamma': [1,0.1,0.01,0.001,0.0001,0.00001],
                'degree': [0,1],'kernel': ['poly']} 


# In[ ]:


param_rbf = {'C': [100, 1000,10000,100000,1000000],'degree':[0,1,2], 'gamma': [0.01,0.001,0.0001,0.00001,0.000001], 'kernel': ['rbf']} 


# In[19]:


poly_model=SVC(kernel='poly',degree=1,C=100,gamma=0.00001)


# In[20]:


poly_model.fit(X_train,y_train)


# In[21]:


grid_poly = GridSearchCV(SVC(),param_poly,refit=True,verbose=3)


# In[2]:


grid_poly.fit(X_train,y_train)


# In[3]:


grid_poly.best_params_


# In[15]:


grid_poly.best_estimator_


# In[17]:


grid_poly_pred = grid_poly.predict(X_test)


# In[18]:


from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,grid_poly_pred))
print(confusion_matrix(y_test,grid_poly_pred))


# # Decision tree & random forest

# ## Tree

# In[62]:


from sklearn.tree import DecisionTreeClassifier


# In[63]:


dtree = DecisionTreeClassifier()


# In[64]:


dtree.fit(X_train,y_train)


# In[65]:


tree_pred = dtree.predict(X_test)


# In[66]:


print(classification_report(y_test,tree_pred))
print(confusion_matrix(y_test,tree_pred))
print('\n','accuracy')
print(accuracy_score(y_test,tree_pred))


# ## Forest

# In[90]:


from sklearn.ensemble import RandomForestClassifier


# In[91]:


rfc = RandomForestClassifier(n_estimators=300)


# In[92]:


rfc.fit(X_train, y_train)


# In[93]:


rfc_pred = rfc.predict(X_test)


# In[80]:


# print(classification_report(y_test,rfc_pred))
# print(confusion_matrix(y_test,rfc_pred))
# print('\n','accuracy')
# print(accuracy_score(y_test,rfc_pred))


# In[96]:


from sklearn.model_selection import GridSearchCV


# In[115]:


# param_rfc = {'n_estimators': [10,200,300,600],
#              #[10,50,100,150,200,300,400,500,600]
#              #np.linspace(0.1, 1.0, 10, endpoint=True)'max_depth': [n for n in range(0,40)],
#              'min_samples_split': [0.1,0.3,0.5,0.7,0.9],
#              #[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
#              'min_samples_leaf': [0.1,0.3,0.5]}

param_rfc = dict(     
    max_depth = [n for n in range(9, 14)],     
    min_samples_split = [n for n in range(4, 11)], 
    min_samples_leaf = [n for n in range(2, 5)],     
    n_estimators = [n for n in range(10, 300, 10)],
)


# In[116]:


grid_rfc = GridSearchCV(estimator=rfc, param_grid=param_rfc, cv= 5,refit=True,verbose=3)


# In[4]:


grid_rfc.fit(X_train,y_train)


# In[119]:


grid_rfc.best_params_


# In[120]:


grid_rfc.best_estimator_


# In[123]:


grid_rfc_pred = grid_rfc.predict(X_test)


# In[122]:


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print(classification_report(y_test,grid_rfc_pred))
print(confusion_matrix(y_test,grid_rfc_pred))
print('\n','accuracy')
print(accuracy_score(y_test,grid_rfc_pred))


# ## KNN

# In[103]:


from sklearn.neighbors import KNeighborsClassifier


# In[126]:


knn = KNeighborsClassifier(n_neighbors=1)


# In[127]:


knn.fit(X_train,y_train)


# In[128]:


knn_pred = knn.predict(X_test)


# In[129]:


print(classification_report(y_test,knn_pred))
print(confusion_matrix(y_test,knn_pred))
print('\n','accuracy')
print(accuracy_score(y_test,knn_pred))


# In[130]:


error_rate = []

# Will take some time
for i in range(1,60):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# In[131]:


plt.figure(figsize=(10,6))
plt.plot(range(1,60),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# ## Output prediction results

# In[124]:


# prediction = pd.DataFrame(rfc_pred, columns=['rfc_pred']).to_csv('prediction.csv')

prediction = pd.DataFrame(grid_rfc_pred, columns=['grid_rfc_pred']).to_csv('prediction.csv')


# In[114]:


# result = X_test.append(prediction)


# In[348]:


# result.to_csv('result.csv')

