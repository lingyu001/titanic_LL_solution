
# Titanic LL solution

#### Attempt the solution of Titanic Data from https://www.kaggle.com/c/titanic using Logistic Regression, Regression Tree and SVM.

## Import libs


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

## Explore the data


```python
tt_train = pd.read_csv('train.csv')
tt_test = pd.read_csv('test.csv')
```


```python
# tt = tt_train.append(tt_test, ignore_index=True)
tt = pd.concat([tt_train,tt_test])
tt = tt.set_index('PassengerId')
```

    C:\Users\liuxi\Anaconda3\lib\site-packages\ipykernel_launcher.py:2: FutureWarning: Sorting because non-concatenation axis is not aligned. A future version
    of pandas will change to not sort by default.
    
    To accept the future behavior, pass 'sort=False'.
    
    To retain the current behavior and silence the warning, pass 'sort=True'.
    
      
    


```python
tt.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1309 entries, 1 to 1309
    Data columns (total 11 columns):
    Age         1046 non-null float64
    Cabin       295 non-null object
    Embarked    1307 non-null object
    Fare        1308 non-null float64
    Name        1309 non-null object
    Parch       1309 non-null int64
    Pclass      1309 non-null int64
    Sex         1309 non-null object
    SibSp       1309 non-null int64
    Survived    891 non-null float64
    Ticket      1309 non-null object
    dtypes: float64(3), int64(3), object(5)
    memory usage: 122.7+ KB
    


```python
tt_train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 12 columns):
    PassengerId    891 non-null int64
    Survived       891 non-null int64
    Pclass         891 non-null int64
    Name           891 non-null object
    Sex            891 non-null object
    Age            714 non-null float64
    SibSp          891 non-null int64
    Parch          891 non-null int64
    Ticket         891 non-null object
    Fare           891 non-null float64
    Cabin          204 non-null object
    Embarked       889 non-null object
    dtypes: float64(2), int64(5), object(5)
    memory usage: 83.6+ KB
    


```python
tt.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Fare</th>
      <th>Name</th>
      <th>Parch</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>SibSp</th>
      <th>Survived</th>
      <th>Ticket</th>
    </tr>
    <tr>
      <th>PassengerId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>22.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>7.2500</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>1</td>
      <td>0.0</td>
      <td>A/5 21171</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38.0</td>
      <td>C85</td>
      <td>C</td>
      <td>71.2833</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>0</td>
      <td>1</td>
      <td>female</td>
      <td>1</td>
      <td>1.0</td>
      <td>PC 17599</td>
    </tr>
    <tr>
      <th>3</th>
      <td>26.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>7.9250</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>0</td>
      <td>3</td>
      <td>female</td>
      <td>0</td>
      <td>1.0</td>
      <td>STON/O2. 3101282</td>
    </tr>
    <tr>
      <th>4</th>
      <td>35.0</td>
      <td>C123</td>
      <td>S</td>
      <td>53.1000</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>0</td>
      <td>1</td>
      <td>female</td>
      <td>1</td>
      <td>1.0</td>
      <td>113803</td>
    </tr>
    <tr>
      <th>5</th>
      <td>35.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>8.0500</td>
      <td>Allen, Mr. William Henry</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>0</td>
      <td>0.0</td>
      <td>373450</td>
    </tr>
  </tbody>
</table>
</div>




```python
tt_train.describe()
```


```python
#tt_train.info()
tt.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1309 entries, 0 to 417
    Data columns (total 12 columns):
    Age            1046 non-null float64
    Cabin          295 non-null object
    Embarked       1307 non-null object
    Fare           1308 non-null float64
    Name           1309 non-null object
    Parch          1309 non-null int64
    PassengerId    1309 non-null int64
    Pclass         1309 non-null int64
    Sex            1309 non-null object
    SibSp          1309 non-null int64
    Survived       891 non-null float64
    Ticket         1309 non-null object
    dtypes: float64(3), int64(4), object(5)
    memory usage: 132.9+ KB
    


```python
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=tt_train)
```


```python
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=tt_train)
```


```python
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Embarked',data=tt_train)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2ca947dbfd0>




![png](output_14_1.png)



```python
plt.figure(figsize=(10,6))
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='SibSp',data=tt_train,)
plt.legend(loc='upper right')
```


```python
plt.figure(figsize=(10,6))
#sns.distplot(tt_train['Age'].dropna(),hue='Survived',kde=False,color='blue',bins=30)
plt.hist(tt_train['Age'][tt_train['Survived']==0],bins=25, alpha=0.5, label='non-survived', color='red')
plt.hist(tt_train['Age'][tt_train['Survived']==1],bins=25, alpha=0.5, label='survived', color='blue')
```




    (array([20., 15.,  5.,  4., 13., 43., 19., 28., 39., 36., 24., 17., 26.,
            10., 11., 11.,  7.,  3.,  5.,  4.,  0.,  0.,  0.,  0.,  1.]),
     array([ 0.42  ,  3.6032,  6.7864,  9.9696, 13.1528, 16.336 , 19.5192,
            22.7024, 25.8856, 29.0688, 32.252 , 35.4352, 38.6184, 41.8016,
            44.9848, 48.168 , 51.3512, 54.5344, 57.7176, 60.9008, 64.084 ,
            67.2672, 70.4504, 73.6336, 76.8168, 80.    ]),
     <a list of 25 Patch objects>)




![png](output_16_1.png)



```python
plt.figure(figsize=(10,6))
sns.pairplot(tt_train['Fare'].dropna(),kde=False,color='blue',bins=30)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2ca940a98d0>




![png](output_17_1.png)



```python
tt_train['Parch'].value_counts()
```




    0    678
    1    118
    2     80
    5      5
    3      5
    4      4
    6      1
    Name: Parch, dtype: int64



## Check Missing data


```python
#sns.heatmap(tt_train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.heatmap(tt_test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1fdda792780>




![png](output_20_1.png)



```python
# Check Age
plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=tt_train,palette='winter')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1e2769aa4e0>




![png](output_21_1.png)



```python
# Check Cabin
#cabin_index = tt_train['Cabin'].astype(str).str[0]
cabin_index = tt_train['Cabin'].astype(str).str[0]
```


```python
sns.set_style('whitegrid')
# sns.countplot(x='Survived',hue=cabin_index,data=tt_train)
sns.countplot(x='Survived',hue=cabin_index,data=tt)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1e27bbcbbe0>




![png](output_23_1.png)



```python
#tt_train.groupby(cabin_index).mean()
tt.groupby(cabin_index).mean()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Fare</th>
      <th>Parch</th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>SibSp</th>
      <th>Survived</th>
    </tr>
    <tr>
      <th>Cabin</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>44.833333</td>
      <td>39.623887</td>
      <td>0.133333</td>
      <td>439.600000</td>
      <td>1.000000</td>
      <td>0.133333</td>
      <td>0.466667</td>
    </tr>
    <tr>
      <th>B</th>
      <td>34.955556</td>
      <td>113.505764</td>
      <td>0.574468</td>
      <td>521.808511</td>
      <td>1.000000</td>
      <td>0.361702</td>
      <td>0.744681</td>
    </tr>
    <tr>
      <th>C</th>
      <td>36.086667</td>
      <td>100.151341</td>
      <td>0.474576</td>
      <td>406.440678</td>
      <td>1.000000</td>
      <td>0.644068</td>
      <td>0.593220</td>
    </tr>
    <tr>
      <th>D</th>
      <td>39.032258</td>
      <td>57.244576</td>
      <td>0.303030</td>
      <td>475.939394</td>
      <td>1.121212</td>
      <td>0.424242</td>
      <td>0.757576</td>
    </tr>
    <tr>
      <th>E</th>
      <td>38.116667</td>
      <td>46.026694</td>
      <td>0.312500</td>
      <td>502.437500</td>
      <td>1.312500</td>
      <td>0.312500</td>
      <td>0.750000</td>
    </tr>
    <tr>
      <th>F</th>
      <td>19.954545</td>
      <td>18.696792</td>
      <td>0.538462</td>
      <td>370.384615</td>
      <td>2.384615</td>
      <td>0.538462</td>
      <td>0.615385</td>
    </tr>
    <tr>
      <th>G</th>
      <td>14.750000</td>
      <td>13.581250</td>
      <td>1.250000</td>
      <td>216.000000</td>
      <td>3.000000</td>
      <td>0.500000</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>T</th>
      <td>45.000000</td>
      <td>35.500000</td>
      <td>0.000000</td>
      <td>340.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>n</th>
      <td>27.555293</td>
      <td>19.157325</td>
      <td>0.365357</td>
      <td>443.208151</td>
      <td>2.639010</td>
      <td>0.547307</td>
      <td>0.299854</td>
    </tr>
  </tbody>
</table>
</div>



## Extract Title from Name


```python
# tt_train['Title'] = tt_train['Name'].str.split(',').apply(lambda x: x[1])
# title = tt_train['Name'].str.split(',').apply(lambda x: x[1])
title = tt['Name'].str.split(',').apply(lambda x: x[1])
```


```python
title = title.str.split('.').apply(lambda x: x[0])
```


```python
title.value_counts()
```




     Mr              757
     Miss            260
     Mrs             197
     Master           61
     Rev               8
     Dr                8
     Col               4
     Major             2
     Ms                2
     Mlle              2
     Don               1
     Jonkheer          1
     Sir               1
     Lady              1
     Capt              1
     Mme               1
     Dona              1
     the Countess      1
    Name: Name, dtype: int64




```python
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
```


```python
title_update = title.map(title_cat)
```


```python
title_update.isna().sum()
```




    0




```python
tt[title_update.isna()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Fare</th>
      <th>Name</th>
      <th>Parch</th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>SibSp</th>
      <th>Survived</th>
      <th>Ticket</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue=title_update,data=tt_train)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2ca94377198>




![png](output_33_1.png)



```python
title_update.value_counts()
```




    1309



## Age Imputation


```python
# tt_train['Age'].groupby(tt_train['Pclass']).median()
tt['Age'].groupby(tt['Pclass']).median()
```




    Pclass
    1    39.0
    2    29.0
    3    24.0
    Name: Age, dtype: float64




```python
tt_train['Age'].groupby(tt_train['Pclass']).mean()
```




    Pclass
    1    38.233441
    2    29.877630
    3    25.140620
    Name: Age, dtype: float64




```python
tt_test['Age'].groupby(tt_test['Pclass']).mean()
```




    Pclass
    1    40.918367
    2    28.777500
    3    24.027945
    Name: Age, dtype: float64




```python
#tt_train['Age'].groupby([tt_train['Sex'],tt_train['Pclass'],title_update]).median()
tt['Age'].groupby([tt['Sex'],tt['Pclass'],title_update]).median()
```




    Sex     Pclass  Name   
    female  1       Miss       30.0
                    Mrs        45.0
                    Noble      39.0
                    Officer    49.0
            2       Miss       20.0
                    Mrs        30.5
            3       Miss       18.0
                    Mrs        31.0
    male    1       Master      6.0
                    Mr         41.5
                    Noble      40.0
                    Officer    52.0
            2       Master      2.0
                    Mr         30.0
                    Officer    41.5
            3       Master      6.0
                    Mr         26.0
    Name: Age, dtype: float64




```python
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
```


```python
# Imputate Age
#tt_train['Age'] = tt_train[['Age','Pclass']].apply(impute_age,axis=1)
```


```python
# Imputation by pclass
# pclass_group = tt_train.groupby('Pclass')
pclass_group = tt.groupby('Pclass')
```


```python
pclass_group.Age.median()
```




    Pclass
    1    39.0
    2    29.0
    3    24.0
    Name: Age, dtype: float64




```python
# tt_train['Age'] = pclass_group.Age.apply(lambda x: x.fillna(x.median()))
tt['Age'] = pclass_group.Age.apply(lambda x: x.fillna(x.median()))
```


```python
#Imputation by sex,pclass and title
# age_group = tt_train.groupby(['Sex','Pclass', title_update]) 
age_group = tt.groupby(['Sex','Pclass', title_update]) 
```


```python
# tt_train['Age'] = age_group.Age.apply(lambda x: x.fillna(x.median()))
tt['Age'] = age_group.Age.apply(lambda x: x.fillna(x.median()))
```


```python
# tt_train.Age.isna().sum()
tt['Age'].isna().sum()
```




    0




```python
tt[tt['Age'].isna()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Fare</th>
      <th>Name</th>
      <th>Parch</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>SibSp</th>
      <th>Survived</th>
      <th>Ticket</th>
    </tr>
    <tr>
      <th>PassengerId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



## Cabin Imputation


```python
# Imputate Carbin, grab first letter
# cabin_index = tt_train.Cabin.astype(str).str[0]
cabin_index = tt.Cabin.astype(str).str[0]
```


```python
cabin_index.value_counts()
```




    n    1014
    C      94
    B      65
    D      46
    E      41
    A      22
    F      21
    G       5
    T       1
    Name: Cabin, dtype: int64



## Fare Imputation


```python
 # Impute with the median Fare for the same Pclass
tt['Fare'][tt['Fare'].isna()] = tt['Fare'][tt['Pclass']==3].median()
```

    C:\Users\liuxi\Anaconda3\lib\site-packages\ipykernel_launcher.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      
    


```python
tt[tt['Fare'].isna()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Fare</th>
      <th>Name</th>
      <th>Parch</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>SibSp</th>
      <th>Survived</th>
      <th>Ticket</th>
    </tr>
    <tr>
      <th>PassengerId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



## Embarked imputation by adjacent ticket number


```python
tt['Embarked'][tt['Embarked'].isna()] = 'S'
```

    C:\Users\liuxi\Anaconda3\lib\site-packages\ipykernel_launcher.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      """Entry point for launching an IPython kernel.
    

## Create dummies for categorical variables

    1. Sex: convert to male, female, create dummy
    2. Children: Age < 16
    2. Family size: create category according to Sibsip and Parch
    3. Pclass: create dummy
    4. Embark: create dummy
    5. Cabin index: create dummy


```python
#Sex dummy
# sex = pd.get_dummies(tt_train['Sex'],drop_first=True)
sex = pd.get_dummies(tt['Sex'],drop_first=True)
```


```python
sex.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>male</th>
    </tr>
    <tr>
      <th>PassengerId</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Children dummy
# child = pd.get_dummies(tt_train['Age'] < 16,prefix='child',drop_first=True)
child = pd.get_dummies(tt['Age'] < 16,prefix='child',drop_first=True)
```


```python
child.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>child_True</th>
    </tr>
    <tr>
      <th>PassengerId</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Combine Child and Sex variable
sex['male'][child['child_True']==1]= 0
```


```python
# Family size
# familysize = 1 + tt_train['Parch'] + tt_train['SibSp']

tt['familysize'] = 1 + tt['Parch'] + tt['SibSp']
```


```python
tt['familysize'].head()
```




    PassengerId
    1    2
    2    2
    3    1
    4    2
    5    1
    Name: familysize, dtype: int64




```python
# Pclass dummy
# pclass = pd.get_dummies(tt_train['Pclass'],prefix='pclass',drop_first=True)
pclass = pd.get_dummies(tt['Pclass'],prefix='pclass',drop_first=True)
```


```python
pclass.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pclass_2</th>
      <th>pclass_3</th>
    </tr>
    <tr>
      <th>PassengerId</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Embark dummy
# embark = pd.get_dummies(tt_train['Embarked'],prefix='embark',drop_first=True)
embark = pd.get_dummies(tt['Embarked'],prefix='embark',drop_first=True)
```


```python
# Cabin index dummy
cabin = pd.get_dummies(cabin_index,prefix='cabin',drop_first=True)
```


```python
cabin.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cabin_B</th>
      <th>cabin_C</th>
      <th>cabin_D</th>
      <th>cabin_E</th>
      <th>cabin_F</th>
      <th>cabin_G</th>
      <th>cabin_T</th>
      <th>cabin_n</th>
    </tr>
    <tr>
      <th>PassengerId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



### Produce the analytic data


```python
tt.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Fare</th>
      <th>Name</th>
      <th>Parch</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>SibSp</th>
      <th>Survived</th>
      <th>Ticket</th>
      <th>familysize</th>
    </tr>
    <tr>
      <th>PassengerId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>22.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>7.2500</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>1</td>
      <td>0.0</td>
      <td>A/5 21171</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38.0</td>
      <td>C85</td>
      <td>C</td>
      <td>71.2833</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>0</td>
      <td>1</td>
      <td>female</td>
      <td>1</td>
      <td>1.0</td>
      <td>PC 17599</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>26.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>7.9250</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>0</td>
      <td>3</td>
      <td>female</td>
      <td>0</td>
      <td>1.0</td>
      <td>STON/O2. 3101282</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>35.0</td>
      <td>C123</td>
      <td>S</td>
      <td>53.1000</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>0</td>
      <td>1</td>
      <td>female</td>
      <td>1</td>
      <td>1.0</td>
      <td>113803</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>35.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>8.0500</td>
      <td>Allen, Mr. William Henry</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>0</td>
      <td>0.0</td>
      <td>373450</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# tt_train = pd.concat([tt_train['Survived'],tt_train['Age'],tt_train['Fare'],sex,child,familysize,pclass,embark,cabin],axis=1)
tt = pd.concat([tt['Survived'],tt['Age'],tt['Fare'],sex,child,tt['familysize'],
                tt['Parch'],tt['SibSp'],pclass,embark,cabin],axis=1)

```


```python
tt['male']=sex['male'].values
```


```python
pd.crosstab(tt['male'],tt['child_True'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>child_True</th>
      <th>0</th>
      <th>1</th>
    </tr>
    <tr>
      <th>male</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>410</td>
      <td>123</td>
    </tr>
    <tr>
      <th>1</th>
      <td>776</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#tt_train.to_csv('tt_train.csv') 
#tt_train_2 = pd.read_csv('tt_train.csv')
```


```python
#tt_train = pd.concat([tt_train,title_dummy],axis=1)
```


```python
from sklearn.model_selection import train_test_split
```


```python
X = tt_train.drop('Survived',axis=1)
y = tt_train['Survived']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=101)
```


```python
X_train = tt[tt['Survived'].notnull()].drop('Survived',axis=1)
y_train = tt[tt['Survived'].notnull()]['Survived']
X_test = tt[tt['Survived'].isna()].drop('Survived',axis=1)
y_test = tt[tt['Survived'].isna()]['Survived']
```


```python
X_train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 891 entries, 1 to 891
    Data columns (total 19 columns):
    Age           891 non-null float64
    Fare          891 non-null float64
    male          891 non-null uint8
    child_True    891 non-null uint8
    familysize    891 non-null int64
    pclass_2      891 non-null uint8
    pclass_3      891 non-null uint8
    embark_Q      891 non-null uint8
    embark_S      891 non-null uint8
    cabin_B       891 non-null uint8
    cabin_C       891 non-null uint8
    cabin_D       891 non-null uint8
    cabin_E       891 non-null uint8
    cabin_F       891 non-null uint8
    cabin_G       891 non-null uint8
    cabin_T       891 non-null uint8
    cabin_n       891 non-null uint8
    Parch         891 non-null int64
    SibSp         891 non-null int64
    dtypes: float64(2), int64(3), uint8(14)
    memory usage: 53.9 KB
    


```python
X_test.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 418 entries, 892 to 1309
    Data columns (total 19 columns):
    Age           418 non-null float64
    Fare          418 non-null float64
    male          418 non-null uint8
    child_True    418 non-null uint8
    familysize    418 non-null int64
    pclass_2      418 non-null uint8
    pclass_3      418 non-null uint8
    embark_Q      418 non-null uint8
    embark_S      418 non-null uint8
    cabin_B       418 non-null uint8
    cabin_C       418 non-null uint8
    cabin_D       418 non-null uint8
    cabin_E       418 non-null uint8
    cabin_F       418 non-null uint8
    cabin_G       418 non-null uint8
    cabin_T       418 non-null uint8
    cabin_n       418 non-null uint8
    Parch         418 non-null int64
    SibSp         418 non-null int64
    dtypes: float64(2), int64(3), uint8(14)
    memory usage: 25.3 KB
    


```python
y_train.head()
```




    PassengerId
    1    0.0
    2    1.0
    3    1.0
    4    1.0
    5    0.0
    Name: Survived, dtype: float64



## Logistic Regression


```python
from sklearn.linear_model import LogisticRegression
```


```python
logit_model = LogisticRegression()
logit_model.fit(X_train,y_train)
```

    C:\Users\liuxi\Anaconda3\lib\site-packages\sklearn\linear_model\logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='warn',
              n_jobs=None, penalty='l2', random_state=None, solver='warn',
              tol=0.0001, verbose=0, warm_start=False)




```python
logit_pred = logit_model.predict(X_test)
```


```python
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print(classification_report(y_test,logit_pred))
print(confusion_matrix(y_test,logit_pred))
print('\n','accuracy')
print(accuracy_score(y_test,logit_pred))
```

                  precision    recall  f1-score   support
    
               0       0.77      0.88      0.82       127
               1       0.81      0.66      0.72        96
    
       micro avg       0.78      0.78      0.78       223
       macro avg       0.79      0.77      0.77       223
    weighted avg       0.79      0.78      0.78       223
    
    [[112  15]
     [ 33  63]]
    
     accuracy
    0.7847533632286996
    

# SVM 


```python
from sklearn.svm import SVC
```


```python
svm_model=SVC(kernel='rbf',C=1000000,gamma=0.00001)
#'linear','poly','rbf','sigmoid','precomputed'
```


```python
svm_model.fit(X_train,y_train)
```




    SVC(C=1000000, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma=1e-05, kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)




```python
svm_pred = svm_model.predict(X_test)
```


```python
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,svm_pred))
print(confusion_matrix(y_test,svm_pred))
print('\n','accuracy')
print(accuracy_score(y_test,svm_pred))
```

                  precision    recall  f1-score   support
    
               0       0.77      0.85      0.81       127
               1       0.77      0.67      0.72        96
    
       micro avg       0.77      0.77      0.77       223
       macro avg       0.77      0.76      0.76       223
    weighted avg       0.77      0.77      0.77       223
    
    [[108  19]
     [ 32  64]]
    
     accuracy
    0.7713004484304933
    

### Gridsearch


```python
from sklearn.model_selection import GridSearchCV
```

#### SVM with rbf kernel


```python
param_rbf = {'C': [100, 1000,10000,100000,1000000], 'gamma': [0.01,0.001,0.0001,0.00001,0.000001], 'kernel': ['rbf']} 
#'linear','poly','rbf','sigmoid','precomputed'
```


```python
grid_rbf = GridSearchCV(SVC(),param_rbf,refit=True,verbose=3)
```


```python
grid_rbf.fit(X_train,y_train)
```


```python
grid_rbf.best_params_
```




    {'C': 1000000, 'gamma': 1e-05, 'kernel': 'rbf'}




```python
grid_rbf.best_estimator_
```




    SVC(C=1000000, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma=1e-05, kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)




```python
grid_rbf_pred = grid_rbf.predict(X_test)
```


```python
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,grid_rbf_pred))
print(confusion_matrix(y_test,grid_rbf_pred))
print('\n','accuracy')
print(accuracy_score(y_test,grid_rbf_pred))
```

                  precision    recall  f1-score   support
    
               0       0.77      0.86      0.81       127
               1       0.78      0.67      0.72        96
    
       micro avg       0.78      0.78      0.78       223
       macro avg       0.78      0.76      0.77       223
    weighted avg       0.78      0.78      0.77       223
    
    [[109  18]
     [ 32  64]]
    
     accuracy
    0.7757847533632287
    

#### SVM with poly kernel


```python
#param_poly = {'C': [100],'gamma': [1,0.1],
             # 'degree': [1],'kernel': ['poly']} 
#param_poly = {'C': [0.1,1, 10, 100, 1000,10000,100000],'gamma': [100,10,1,0.1,0.01,0.001,0.0001,0.00001,0.000001],
     #         'degree': [0,1,2,3,4,5,6],'kernel': ['poly']} 
param_poly = {'C': [0.1,1, 10, 100, 1000,10000,100000],'gamma': [1,0.1,0.01,0.001,0.0001,0.00001],
                'degree': [0,1],'kernel': ['poly']} 

```


```python
param_rbf = {'C': [100, 1000,10000,100000,1000000],'degree':[0,1,2], 'gamma': [0.01,0.001,0.0001,0.00001,0.000001], 'kernel': ['rbf']} 
```


```python
poly_model=SVC(kernel='poly',degree=1,C=100,gamma=0.00001)
```


```python
poly_model.fit(X_train,y_train)
```




    SVC(C=100, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=1, gamma=1e-05, kernel='poly',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)




```python
grid_poly = GridSearchCV(SVC(),param_poly,refit=True,verbose=3)
```


```python
grid_poly.fit(X_train,y_train)
```


```python
grid_poly.best_params_
```


```python
grid_poly.best_estimator_
```




    SVC(C=100, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=1, gamma=0.1, kernel='poly',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)




```python
grid_poly_pred = grid_poly.predict(X_test)
```


```python
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,grid_poly_pred))
print(confusion_matrix(y_test,grid_poly_pred))
```

                  precision    recall  f1-score   support
    
               0       0.77      0.88      0.82       154
               1       0.80      0.65      0.72       114
    
       micro avg       0.78      0.78      0.78       268
       macro avg       0.79      0.77      0.77       268
    weighted avg       0.79      0.78      0.78       268
    
    [[136  18]
     [ 40  74]]
    

# Decision tree & random forest

## Tree


```python
from sklearn.tree import DecisionTreeClassifier
```


```python
dtree = DecisionTreeClassifier()
```


```python
dtree.fit(X_train,y_train)
```




    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                splitter='best')




```python
tree_pred = dtree.predict(X_test)
```


```python
print(classification_report(y_test,tree_pred))
print(confusion_matrix(y_test,tree_pred))
print('\n','accuracy')
print(accuracy_score(y_test,tree_pred))
```

                  precision    recall  f1-score   support
    
               0       0.80      0.87      0.84       127
               1       0.81      0.72      0.76        96
    
       micro avg       0.81      0.81      0.81       223
       macro avg       0.81      0.80      0.80       223
    weighted avg       0.81      0.81      0.81       223
    
    [[111  16]
     [ 27  69]]
    
     accuracy
    0.8071748878923767
    

## Forest


```python
from sklearn.ensemble import RandomForestClassifier
```


```python
rfc = RandomForestClassifier(n_estimators=300)
```


```python
rfc.fit(X_train, y_train)
```




    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=300, n_jobs=None,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False)




```python
rfc_pred = rfc.predict(X_test)
```


```python
# print(classification_report(y_test,rfc_pred))
# print(confusion_matrix(y_test,rfc_pred))
# print('\n','accuracy')
# print(accuracy_score(y_test,rfc_pred))
```


```python
from sklearn.model_selection import GridSearchCV
```


```python
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
```


```python
grid_rfc = GridSearchCV(estimator=rfc, param_grid=param_rfc, cv= 5,refit=True,verbose=3)
```


```python
grid_rfc.fit(X_train,y_train)
```


```python
grid_rfc.best_params_
```




    {'max_depth': 12,
     'min_samples_leaf': 3,
     'min_samples_split': 10,
     'n_estimators': 20}




```python
grid_rfc.best_estimator_
```




    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=12, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=3, min_samples_split=10,
                min_weight_fraction_leaf=0.0, n_estimators=20, n_jobs=None,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False)




```python
grid_rfc_pred = grid_rfc.predict(X_test)
```


```python
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print(classification_report(y_test,grid_rfc_pred))
print(confusion_matrix(y_test,grid_rfc_pred))
print('\n','accuracy')
print(accuracy_score(y_test,grid_rfc_pred))
```

## KNN


```python
from sklearn.neighbors import KNeighborsClassifier
```


```python
knn = KNeighborsClassifier(n_neighbors=1)
```


```python
knn.fit(X_train,y_train)
```




    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
               metric_params=None, n_jobs=None, n_neighbors=1, p=2,
               weights='uniform')




```python
knn_pred = knn.predict(X_test)
```


```python
print(classification_report(y_test,knn_pred))
print(confusion_matrix(y_test,knn_pred))
print('\n','accuracy')
print(accuracy_score(y_test,knn_pred))
```

                  precision    recall  f1-score   support
    
               0       0.71      0.82      0.76       127
               1       0.70      0.55      0.62        96
    
       micro avg       0.70      0.70      0.70       223
       macro avg       0.70      0.69      0.69       223
    weighted avg       0.70      0.70      0.70       223
    
    [[104  23]
     [ 43  53]]
    
     accuracy
    0.7040358744394619
    


```python
error_rate = []

# Will take some time
for i in range(1,60):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
```


```python
plt.figure(figsize=(10,6))
plt.plot(range(1,60),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
```




    Text(0, 0.5, 'Error Rate')




![png](output_144_1.png)


## Output prediction results


```python
# prediction = pd.DataFrame(rfc_pred, columns=['rfc_pred']).to_csv('prediction.csv')

prediction = pd.DataFrame(grid_rfc_pred, columns=['grid_rfc_pred']).to_csv('prediction.csv')
```


```python
# result = X_test.append(prediction)
```


```python
# result.to_csv('result.csv')
```
