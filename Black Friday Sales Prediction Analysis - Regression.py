#!/usr/bin/env python
# coding: utf-8

# ## Dataset Information
# 
# This dataset comprises of sales transactions captured at a retail store. It’s a classic dataset to explore and expand your feature engineering skills and day to day understanding from multiple shopping experiences. This is a regression problem. The dataset has 550,069 rows and 12 columns.
# 
# **Problem:** Predict purchase amount
# 
# ## Attributes:
# | Column ID |         Column Name        | Data type |           Description           | Masked |
# |:---------:|:--------------------------:|:---------:|:-------------------------------:|--------|
# |     0     |           User_ID          |   int64   |      Unique Id of customer      | False  |
# |     1     |         Product_ID         |   object  |       Unique Id of product      | False  |
# |     2     |           Gender           |   object  |         Sex of customer         | False  |
# |     3     |             Age            |   object  |         Age of customer         | False  |
# |     4     |         Occupation         |   int64   |   Occupation code of customer   | True   |
# |     5     |        City_Category       |   object  |         City of customer        | True   |
# |     6     | Stay_In_Current_City_Years |   object  | Number of years of stay in city | False  |
# |     7     |       Marital_Status       |   int64   |    Marital status of customer   | False  |
# |     8     |     Product_Category_1     |   int64   |       Category of product       | True   |
# |     9     |     Product_Category_2     |  float64  |       Category of product       | True   |
# |     10    |     Product_Category_3     |  float64  |       Category of product       | True   |
# |     11    |          Purchase          |   int64   |         Purchase amount         | False  |

# ## Import modules

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')


# ## Loading the dataset

# In[2]:


df = pd.read_csv('train.csv')
df.head()


# In[3]:


# statistical info
df.describe()


# In[4]:


# datatype info
df.info()


# In[5]:


# find unique values
df.apply(lambda x: len(x.unique()))


# ## Exploratory Data Analysis

# In[7]:


# distplot for purchase
plt.style.use('fivethirtyeight')
plt.figure(figsize=(13, 7))
sns.distplot(df['Purchase'], bins=25)


# In[8]:


# distribution of numeric variables
sns.countplot(df['Gender'])


# In[10]:


sns.countplot(df['Age'])


# In[11]:


sns.countplot(df['Marital_Status'])


# In[12]:


sns.countplot(df['Occupation'])


# In[13]:


sns.countplot(df['Product_Category_1'])


# In[14]:


sns.countplot(df['Product_Category_2'])


# In[15]:


sns.countplot(df['Product_Category_3'])


# In[16]:


sns.countplot(df['City_Category'])


# In[17]:


sns.countplot(df['Stay_In_Current_City_Years'])


# In[21]:


# bivariate analysis
occupation_plot = df.pivot_table(index='Occupation', values='Purchase', aggfunc=np.mean)
occupation_plot.plot(kind='bar', figsize=(13, 7))
plt.xlabel('Occupation')
plt.ylabel("Purchase")
plt.title("Occupation and Purchase Analysis")
plt.xticks(rotation=0)
plt.show()


# In[22]:


age_plot = df.pivot_table(index='Age', values='Purchase', aggfunc=np.mean)
age_plot.plot(kind='bar', figsize=(13, 7))
plt.xlabel('Age')
plt.ylabel("Purchase")
plt.title("Age and Purchase Analysis")
plt.xticks(rotation=0)
plt.show()


# In[27]:


gender_plot = df.pivot_table(index='Gender', values='Purchase', aggfunc=np.mean)
gender_plot.plot(kind='bar', figsize=(13, 7))
plt.xlabel('Gender')
plt.ylabel("Purchase")
plt.title("Gender and Purchase Analysis")
plt.xticks(rotation=0)
plt.show()


# ## Preprocessing the dataset

# In[28]:


# check for null values
df.isnull().sum()


# In[29]:


df['Product_Category_2'] = df['Product_Category_2'].fillna(-2.0).astype("float32")
df['Product_Category_3'] = df['Product_Category_3'].fillna(-2.0).astype("float32")


# In[30]:


df.isnull().sum()


# In[31]:


# encoding values using dict
gender_dict = {'F':0, 'M':1}
df['Gender'] = df['Gender'].apply(lambda x: gender_dict[x])
df.head()


# In[32]:


# to improve the metric use one hot encoding
# label encoding
cols = ['Age', 'City_Category', 'Stay_In_Current_City_Years']
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in cols:
    df[col] = le.fit_transform(df[col])
df.head()


# ## Coorelation Matrix
# 
# 

# In[34]:


corr = df.corr()
plt.figure(figsize=(14,7))
sns.heatmap(corr, annot=True, cmap='coolwarm')


# ## Input Split

# In[35]:


df.head()


# In[36]:


X = df.drop(columns=['User_ID', 'Product_ID', 'Purchase'])
y = df['Purchase']


# ## Model Training

# In[41]:


from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
def train(model, X, y):
    # train-test split
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.25)
    model.fit(x_train, y_train)
    
    # predict the results
    pred = model.predict(x_test)
    
    # cross validation
    cv_score = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5)
    cv_score = np.abs(np.mean(cv_score))
    
    print("Results")
    print("MSE:", np.sqrt(mean_squared_error(y_test, pred)))
    print("CV Score:", np.sqrt(cv_score))


# In[42]:


from sklearn.linear_model import LinearRegression
model = LinearRegression(normalize=True)
train(model, X, y)
coef = pd.Series(model.coef_, X.columns).sort_values()
coef.plot(kind='bar', title='Model Coefficients')


# In[43]:


from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
train(model, X, y)
features = pd.Series(model.feature_importances_, X.columns).sort_values(ascending=False)
features.plot(kind='bar', title='Feature Importance')


# In[45]:


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_jobs=-1)
train(model, X, y)
features = pd.Series(model.feature_importances_, X.columns).sort_values(ascending=False)
features.plot(kind='bar', title='Feature Importance')


# In[46]:


from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor(n_jobs=-1)
train(model, X, y)
features = pd.Series(model.feature_importances_, X.columns).sort_values(ascending=False)
features.plot(kind='bar', title='Feature Importance')


# In[ ]:





# In[ ]:


pred = model.predict(x_test)


# In[ ]:


submission = pd.DataFrame()
submission['User_ID'] = x_test['User_ID']
submission['Purchase'] = pred


# In[ ]:


submission.to_csv('submission.csv', index=False)

