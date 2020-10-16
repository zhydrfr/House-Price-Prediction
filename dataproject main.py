#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import Packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from pandas.api.types import CategoricalDtype
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder;
from sklearn.preprocessing import RobustScaler , MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression,LinearRegression,SGDClassifier;
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import Ridge , RidgeCV , Lasso
from sklearn import linear_model 
from scipy import stats
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#read data
train = pd.read_csv('C:/Users/lenovo/Downloads/data project/train_data.csv')
test = pd.read_csv('C:/Users/lenovo/Downloads/data project/test_data.csv')
'''Dimensions of train and test data'''
print('Dimensions of train data:', train.shape)
print('Dimensions of test data:', test.shape)


# In[3]:


#check the decoration
train.columns


# In[4]:


#descriptive statistics summary
train['SalePrice'].describe()


# In[5]:


#histogram and normal probability plot for target Value
sns.distplot(train['SalePrice'], fit=norm);
plt.ylabel('Frequency')
plt.title('SalePrice distribution')
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.ylabel('Frequency')
plt.title('SalePrice distribution')


# In[6]:


# SalePrice transformation
train['SalePrice'] = np.log(train['SalePrice'])


# In[7]:


#transformed histogram and normal probability plot target Value
sns.distplot(train['SalePrice'], fit=norm);
plt.ylabel('Frequency')
plt.title('SalePrice distribution')
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)


# In[8]:


target_variable = train['SalePrice']
train.drop('SalePrice',axis=1,inplace=True)


# In[9]:


#plot missing train data before preproceesing
train_na = (train.isnull().sum() / len(train)) * 100
train_na = train_na.drop(train_na[train_na == 0].index).sort_values(ascending=False)[:80]
missing_data = pd.DataFrame({'Missing Ratio' :train_na})
missing_data.head(80)
f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(y= train_na.index, x=train_na)
plt.xlabel('Percent of missing values', fontsize=15)
plt.ylabel('Features', fontsize=15)
plt.title('Percent missing train data by feature in train dataset', fontsize=15)


# In[10]:


#plot missing test data before preproceesing
test_na = (test.isnull().sum() / len(test)) * 100
test_na = test_na.drop(test_na[test_na == 0].index).sort_values(ascending=False)[:80]
missing_data = pd.DataFrame({'Missing Ratio' :test_na})
missing_data.head(80)
f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(y= test_na.index, x=test_na)
plt.xlabel('Percent of missing values', fontsize=15)
plt.ylabel('Features', fontsize=15)
plt.title('Percent missing test data by feature in train dataset', fontsize=15)


# In[11]:


print('Missing Values in Train:','\n',train.isnull().sum().sort_values(ascending = False).head(19))
print('----------------------------')
print('Missing Values in Test:','\n',test.isnull().sum().sort_values(ascending = False).head(19))


# In[12]:


# Concatenate
df = pd.concat((train, test), axis=0)


# In[13]:


# Differences in data types
print(train.info())


# In[14]:


df.isnull().sum().sort_values(ascending = False).head(10)


# In[15]:


# NA percentage
def showNA(df,perc=0):
    #Percentage of NAN Values 
    NAN = [(c, df[c].isna().mean()*100) for c in train]
    NAN = pd.DataFrame(NAN, columns=["column_name", "percentage"])
    NAN = NAN[NAN.percentage > perc]
    print(NAN.sort_values("percentage", ascending=False))

showNA(df)


# In[16]:


# Preprocessing
df['Alley'].fillna(value='No',inplace=True)    
df['BsmtQual'].fillna(value='No',inplace=True)
df['BsmtCond'].fillna(value='No',inplace=True)
df['BsmtExposure'].fillna(value='No',inplace=True)
df['BsmtFinType1'].fillna(value='No',inplace=True)    
df['BsmtFinType2'].fillna(value='No',inplace=True)    
df['FireplaceQu'].fillna(value='No',inplace=True)    
df['GarageType'].fillna(value='No',inplace=True)  
df['GarageFinish'].fillna(value='No',inplace=True)
df['GarageQual'].fillna(value='No',inplace=True)
df['GarageCond'].fillna(value='No',inplace=True)
df['MasVnrType'].fillna(value='No',inplace=True)
df['PoolQC'].fillna(value='No',inplace=True)    
df['Fence'].fillna(value='No',inplace=True)
df['MiscFeature'].fillna(value='No',inplace=True)


# In[17]:


showNA(df)


# In[18]:


# Preprocessing - Continue
df['GarageYrBlt']=df['GarageYrBlt'].fillna(df['GarageYrBlt'].median())
df['MasVnrArea']=df['MasVnrArea'].fillna(df['MasVnrArea'].median())
df['Electrical']=df['Electrical'].fillna(method='ffill')
df['SaleType']=df['SaleType'].fillna(method='ffill')
df['KitchenQual']=df['KitchenQual'].fillna(method='ffill')
df['Exterior1st']=df['Exterior1st'].fillna(method='ffill')
df['Exterior2nd']=df['Exterior2nd'].fillna(method='ffill')
df['Functional']=df['Functional'].fillna(method='ffill')
df['Utilities']=df['Utilities'].fillna(method='ffill')
df['MSZoning']=df['MSZoning'].fillna(method='ffill')
df['GarageCars'] = df['GarageCars'].fillna(df['GarageCars'].median())
df['GarageArea'] = df['GarageArea'].fillna(df['GarageArea'].median())
df['BsmtFinSF1'] = df['BsmtFinSF1'].fillna(df['BsmtFinSF1'].median())
df['TotalBsmtSF'] = df['TotalBsmtSF'].fillna(df['TotalBsmtSF'].median())
df['BsmtUnfSF'] = df['BsmtUnfSF'].fillna(df['BsmtUnfSF'].median())
df['BsmtFinSF2'] = df['BsmtFinSF2'].fillna(df['BsmtFinSF2'].median())
df['BsmtHalfBath'] = df['BsmtHalfBath'].fillna(df['BsmtHalfBath'].median())
df['BsmtFullBath'] = df['BsmtFullBath'].fillna(df['BsmtFullBath'].median())
df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())


# In[19]:


print('Remained Missing Values in Dataset:','\n',df.isnull().sum().sort_values(ascending = False))


# In[20]:


# Infos
df.info()


# In[21]:


df['MoSold'].astype('O')


# In[22]:


df['MSSubClass'].unique()
df['MSSubClass'] = df.replace({'MSSubClass': {20: 'SubClass_20', 30: 'SubClass_30',40: 'SubClass_40',
45: 'SubClass_45',50: 'SubClass_50',60: 'SubClass_60',70: 'SubClass_70',
75: 'SubClass_75',80: 'SubClass_80',85: 'SubClass_85',90: 'SubClass_90',
120: 'SubClass_120',150: 'SubClass_150',160: 'SubClass_160',180: 'SubClass_180',
190: 'SubClass_190'}})
df['MSSubClass']


# In[23]:


df['YrSold'].dtypes
df['YrSold'] = df['YrSold'].astype('O')
df['YrSold']


# In[24]:


# Ordered
df['BsmtCond']=df['BsmtCond'].astype(CategoricalDtype(categories=['No','Po','Fa','TA','Gd','Ex'],ordered=True))
df["BsmtExposure"] = df["BsmtExposure"].astype(CategoricalDtype(categories=['No','Mn','Av','Gd'],ordered=True))
df["BsmtFinType1"] = df["BsmtFinType1"].astype(CategoricalDtype(categories=['No','Unf','LwQ','Rec','BLQ','ALQ','GLQ'],ordered=True))
df["BsmtFinType2"] = df["BsmtFinType2"].astype(CategoricalDtype(categories=['No','Unf','LwQ','Rec','BLQ','ALQ','GLQ'],ordered=True))
df["BsmtQual"] = df["BsmtQual"].astype(CategoricalDtype(categories=['No','Po','Fa','TA','Gd','Ex'],ordered=True))
df["ExterCond"] = df["ExterCond"].astype(CategoricalDtype(categories=['Po','Fa','TA','Gd','Ex'],ordered=True))
df["ExterQual"] = df["ExterQual"].astype(CategoricalDtype(categories=['Po','Fa','TA','Gd','Ex'],ordered=True))
df["Fence"] = df["Fence"].astype(CategoricalDtype(categories=['No','MnWw','GdWo','MnPrv','GdPrv'],ordered=True))
df["FireplaceQu"] = df["FireplaceQu"].astype(CategoricalDtype(categories=['No','Po','Fa','TA','Gd','Ex'],ordered=True))
df["Functional"] = df["Functional"].astype(CategoricalDtype(categories=['Sal','Sev','Maj2','Maj1','Mod','Min2','Min1','Typ'],ordered=True))
df["GarageCond"] = df["GarageCond"].astype(CategoricalDtype(categories=['No','Po','Fa','TA','Gd','Ex'],ordered=True))
df["GarageFinish"] = df["GarageFinish"].astype(CategoricalDtype(categories=['No','Unf','RFn','Fin'],ordered=True))
df["GarageQual"] = df["GarageQual"].astype(CategoricalDtype(categories=['No','Po','Fa','TA','Gd','Ex'],ordered=True))
df["HeatingQC"] = df["HeatingQC"].astype(CategoricalDtype(categories=['Po','Fa','TA','Gd','Ex'],ordered=True))
df["KitchenQual"] = df["KitchenQual"].astype(CategoricalDtype(categories=['Po','Fa','TA','Gd','Ex'],ordered=True))
df["PavedDrive"] = df["PavedDrive"].astype(CategoricalDtype(categories=['N','P','Y'],ordered=True))
df["PoolQC"] = df["PoolQC"].astype(CategoricalDtype(categories=['No','Fa','TA','Gd','Ex'],ordered=True))
df["Utilities"] = df["Utilities"].astype(CategoricalDtype(categories=['ELO','NoSeWa','NoSewr','AllPub'],ordered=True))


# In[25]:


df[['MSSubClass','MoSold']].head(5)


# In[26]:


train = df[0:1314]
train = pd.concat([train, target_variable], axis=1)
print('train dimension: ',train.shape)

test = df[1314:]
print('test dimension: ',test.shape)


# In[165]:


#train.to_csv("train_pyrr.csv")


# # after using Cook Distance in R

# In[27]:


train2 = pd.read_csv('C:/Users/lenovo/Downloads/data project/train_rpy.csv')
Y = train2['SalePrice']
train2.drop('SalePrice',axis=1,inplace=True)
print(train2.shape)
print(Y.shape)


# In[28]:


print(train.columns)
print(train2.columns)


# In[29]:


train2=train2.rename(columns = {'X1stFlrSF':'1stFlrSF','X2ndFlrSF': '2ndFlrSF','X3SsnPorch':'3SsnPorch'})
train2.columns


# In[30]:


# Look for outliers
plt.figure(figsize=[20,5])

# 'GrLivArea' plot
plt.subplot(1,2,1)
sns.scatterplot(x = train2['GrLivArea'], y = Y, color = 'purple')
plt.title('GrLivArea')
plt.ylabel('SalePrice')
plt.xlabel('GrLivArea')
plt.grid(b=bool, which='major', axis='both')

# Will remove two values that don't match distribution on 'GrLivArea'. Turns out this is also the two outliers on OverallQual = 10 (see below)
index_to_drop = train2[(train2['GrLivArea']>4000)&(Y>11.7)].index.tolist()
# Remove from training feature set
train2 = train2.drop(index_to_drop,axis=0)
# Remove from training observation set
Y = Y.drop(index_to_drop)
print(train2.shape)
print(Y.shape)

# 'GrLivArea' plot removed
plt.subplot(1,2,2)
sns.scatterplot(x = train2['GrLivArea'], y = Y, color = 'purple')
plt.title('GrLivArea')
plt.ylabel('SalePrice')
plt.xlabel('GrLivArea')
plt.grid(b=bool, which='major', axis='both')


# In[31]:


plt.figure(figsize=[20,5])

# 'GarageArea' plot
plt.subplot(1,2,1)
sns.scatterplot(x = train2['GarageArea'], y = Y, color = 'green')
plt.title('GarageArea')
plt.ylabel('SalePrice')
plt.xlabel('GarageArea')
plt.grid(b=bool, which='major', axis='both')

# Will remove two values that don't match distribution on 'GarageArea' . Turns out this is also the two outliers on OverallQual = 10 (see below)
index_to_drop = train2[(train2['GarageArea']>1200 )&(Y>11)].index.tolist()
# Remove from training feature set
train2 = train2.drop(index_to_drop,axis=0)
# Remove from training observation set
Y = Y.drop(index_to_drop)
print(train2.shape)
print(Y.shape)

# 'GarageArea' plot removed
plt.subplot(1,2,2)
sns.scatterplot(x = train2['GarageArea'], y = Y, color = 'green')
plt.title('GarageArea')
plt.ylabel('SalePrice')
plt.xlabel('GarageArea')
plt.grid(b=bool, which='major', axis='both')


# In[32]:


plt.figure(figsize=[20,5])

# 'TotalBsmtSF'
plt.subplot(1,2,1)
sns.scatterplot(x = train2['TotalBsmtSF'], y = Y, color = 'orange')
plt.title('TotalBsmtSF')
plt.ylabel('SalePrice')
plt.xlabel('TotalBsmtSF')
plt.grid(b=bool, which='major', axis='both')

# Will remove two values that don't match distribution on 'TotalBsmtSF'. Turns out this is also the two outliers on OverallQual = 10 (see below)
index_to_drop = train2[(train2['TotalBsmtSF']>3000)&(Y>12.5)].index.tolist()
# Remove from training feature set
train2 = train2.drop(index_to_drop,axis=0)
# Remove from training observation set
Y = Y.drop(index_to_drop)
print(train2.shape)
print(Y.shape)

# 'TotalBsmtSF' removed
plt.subplot(1,2,2)
sns.scatterplot(x = train2['TotalBsmtSF'], y = Y, color = 'orange')
plt.title('TotalBsmtSF')
plt.ylabel('SalePrice')
plt.xlabel('TotalBsmtSF')
plt.grid(b=bool, which='major', axis='both')


# In[33]:


plt.figure(figsize=[20,5])

# 'MasVnrArea'
plt.subplot(1,2,1)
sns.scatterplot(x = train2['MasVnrArea'], y = Y , color='brown')
plt.title('MasVnrArea')
plt.ylabel('SalePrice')
plt.xlabel('MasVnrArea')
plt.grid(b=bool, which='major', axis='both')

# Will remove two values that don't match distribution on 'MasVnrArea' . Turns out this is also the two outliers on OverallQual = 10 (see below)
index_to_drop = train2[(train2['MasVnrArea']>1200 )&(Y>12)].index.tolist()
# Remove from training feature set
train2 = train2.drop(index_to_drop,axis=0)
# Remove from training observation set
Y = Y.drop(index_to_drop)
print(train2.shape)
print(Y.shape)

# 'MasVnrArea' removed
plt.subplot(1,2,2)
sns.scatterplot(x = train2['MasVnrArea'], y = Y , color='brown')
plt.title('MasVnrArea')
plt.ylabel('SalePrice')
plt.xlabel('MasVnrArea')
plt.grid(b=bool, which='major', axis='both')


# In[34]:


plt.figure(figsize=[20,5])

# '1stFlrSF'
plt.subplot(1,2,1)
sns.scatterplot(x = train2['1stFlrSF'], y = Y , color='black')
plt.title('1stFlrSF')
plt.ylabel('SalePrice')
plt.xlabel('1stFlrSF')
plt.grid(b=bool, which='major', axis='both')

# Will remove two values that don't match distribution on '1stFlrSF' . Turns out this is also the two outliers on OverallQual = 10 (see below)
index_to_drop = train2[(train2['1stFlrSF']>2700 )&(Y>12.5)].index.tolist()
# Remove from training feature set
train2 = train2.drop(index_to_drop,axis=0)
# Remove from training observation set
Y = Y.drop(index_to_drop)
print(train2.shape)
print(Y.shape)

# '1stFlrSF' removed
plt.subplot(1,2,2)
sns.scatterplot(x = train2['1stFlrSF'], y = Y , color='black')
plt.title('1stFlrSF')
plt.ylabel('SalePrice')
plt.xlabel('1stFlrSF')
plt.grid(b=bool, which='major', axis='both')


# In[35]:


print(train2.shape)
print(test.shape)
print(train2.columns)
print(test.columns)
Y = Y
print(Y.shape)


# In[ ]:





# In[36]:


all_data = pd.concat([train2, test], axis=0).reset_index(drop=True)
all_data.shape


# In[37]:


all_data = all_data.drop(['Utilities', 'Street', 'PoolQC'], axis=1)


# In[38]:


all_data.shape


# In[39]:


# Remove any duplicated column names
all_data = all_data.loc[:,~all_data.columns.duplicated()]


# In[40]:


# drop columns from Regression Result in R
all_data = all_data.drop(['GrLivArea', 'TotalBsmtSF'], axis=1)


# In[41]:


from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 'ExterQual', 'ExterCond','HeatingQC',
        'KitchenQual', 'BsmtFinType1', 'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish',
        'LandSlope','LotShape', 'PavedDrive',  'Alley', 'CentralAir', 'MSSubClass', 'OverallCond','YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values)) 
    all_data[c] = lbl.transform(list(all_data[c].values))


# In[42]:


# No. of categoric variables
cat_feats = all_data.dtypes[all_data.dtypes == "object"].tolist()
print(str(len(cat_feats)) + ' categoric features')
# No. of numerical variables
num_feats = all_data.dtypes[all_data.dtypes != "object"].index.tolist()
print(str(len(num_feats)) + ' numeric features')


# In[43]:


# categorical features
all_data_cat = all_data.select_dtypes(include=['object']).columns
all_data_cat = all_data.select_dtypes(include=['object'])
all_data_cat


# In[44]:


# numerical features 
all_data_num = all_data.select_dtypes(exclude = ["object"]).columns
all_data_num = all_data.select_dtypes(exclude = ['object'])
all_data_num


# In[45]:


# create dummy from categoricals
all_data_cat = pd.get_dummies(all_data_cat,drop_first = True)
all_data_cat.shape


# In[46]:


print(all_data_num.shape)
print(all_data_num.columns)
all_data_num


# In[47]:


all_data_num = pd.DataFrame(all_data_num)
all_data_cat = pd.DataFrame(all_data_cat)


# In[48]:


all_data2 = pd.concat([all_data_num,all_data_cat],axis=1).reset_index(drop=True)
print(all_data2.shape)
print(all_data2.columns)


# In[ ]:





# In[49]:


# drop columns from regression in R after dummy
all_data2 = all_data2.drop(['Exterior2nd_CBlock', 'Exterior1st_BrkComm', 'RoofStyle_Shed','Condition2_RRNn','Condition2_PosN','Exterior1st_Stone'], axis=1)


# In[50]:


all_data2.shape


# In[51]:


# drop columns from VIF>60 in R
all_data2 = all_data2.drop(['SaleCondition_Normal', 'SaleType_WD', 'MiscFeature_Shed','MiscFeature_No','GarageType_No','GarageType_Detchd','GarageType_Attchd','Exterior2nd_VinylSd','Exterior2nd_MetalSd','Exterior1st_VinylSd','Exterior1st_MetalSd','RoofStyle_Hip','RoofStyle_Gable'], axis=1)


# In[52]:


print("Find most important features relative to target")
print('train dimension: ',train.shape)
corr = train.corr()
corr.sort_values(['SalePrice'], ascending = False, inplace = True)
print(corr.SalePrice)
#this you can see at the time of heatmap also.


# In[53]:


train.corr()['SalePrice'].sort_values(ascending=False).iloc[1:].plot(kind='bar', figsize=(12, 4), color='b')


# In[54]:


train.iloc[Y]


# In[ ]:





# In[55]:


# Recreate Train and Test      #### for R
train_= all_data2.iloc[:len(Y), :]
test = all_data2.iloc[len(Y):, :]
print( Y.shape, test.shape,train_.shape)


# In[58]:


ridge2 = RidgeCV(alphas = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60])
ridge2.fit(train_,Y)
alpha = ridge2.alpha_

iteration = 10
cv_ridge2 = KFold(shuffle=True, random_state=2, n_splits=10)
for i in range(iteration):
    ridge2 = RidgeCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, 
                          alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,
                          alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4],cv = cv_ridge2)
    ridge2.fit(train_, Y)
    alpha = ridge2.alpha_
print("Best alpha :", alpha)
coef_ridge2 = pd.Series(ridge2.coef_, index = train_.columns)

print("Total Ridge picked " + str(sum(coef_ridge2 != 0)) + " variables and eliminated the other " +  str(sum(coef_ridge2 == 0)) + " variables")
coef_ridge2 = pd.DataFrame(ridge2.coef_, train_.columns, columns=['Coefficient'])
coef_ridge2


# In[60]:


from sklearn.linear_model import LassoCV
lasso2 = LassoCV(alphas = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60])
lasso2.fit(train_,Y)
alpha = lasso2.alpha_

iteration = 10
cv_lasso2 = KFold(shuffle=True, random_state=2, n_splits=10)
for i in range(iteration):
    lasso2 = LassoCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, 
                          alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,
                          alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4],cv = cv_lasso2)
    lasso2.fit(train_, Y)
    alpha = lasso2.alpha_
print("Best alpha :", alpha)
coef_lasso2 = pd.Series(lasso2.coef_, index = train_.columns)

print("Total Lasso picked " + str(sum(coef_lasso2 != 0)) + " variables and eliminated the other " +  str(sum(coef_lasso2 == 0)) + " variables")
coef_lasso2 = pd.DataFrame(lasso2.coef_, train_.columns, columns=['Coefficient'])
coef_lasso2


# In[61]:


from sklearn.linear_model import ElasticNetCV
elnet2 = ElasticNetCV(alphas = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60])
elnet2.fit(train_,Y)
alpha = elnet2.alpha_

iteration = 10
cv_elnet2 = KFold(shuffle=True, random_state=2, n_splits=10)
for i in range(iteration):
    elnet2 = ElasticNetCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, 
                          alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,
                          alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4],cv = cv_elnet2)
    elnet2.fit(train_, Y)
    alpha = elnet2.alpha_
print("Best alpha :", alpha)
coef_elnet2 = pd.Series(elnet2.coef_, index = train_.columns)

print("Total ElasticNet picked " + str(sum(coef_elnet2 != 0)) + " variables and eliminated the other " +  str(sum(coef_elnet2 == 0)) + " variables")
coef_elnet2 = pd.DataFrame(elnet2.coef_, train_.columns, columns=['Coefficient'])
coef_elnet2


# In[62]:


coef_elnet2.to_csv("coef_elnet2.csv")


# In[63]:


all_data2 = all_data2.drop(['BsmtHalfBath', 'LandContour_Low', 'LandContour_Lvl','LotConfig_FR3','Neighborhood_Blueste','Neighborhood_CollgCr','Neighborhood_NPkVill','Neighborhood_SWISU','Neighborhood_SawyerW','Neighborhood_Timber','Condition1_Feedr','Condition1_PosA','Condition1_RRNe','Condition1_RRNn','Condition2_Feedr','Condition2_PosA','Condition2_RRAe','Condition2_RRAn','HouseStyle_1Story','HouseStyle_2.5Fin','HouseStyle_SLvl','RoofStyle_Gambrel','RoofMatl_Membran','RoofMatl_Metal','RoofMatl_Roll','RoofMatl_Tar&Grv','RoofMatl_WdShake','Exterior1st_AsphShn','Exterior1st_CBlock','Exterior1st_CemntBd','Exterior1st_ImStucc','Exterior1st_WdShing','Exterior2nd_AsphShn','Exterior2nd_Brk Cmn','Exterior2nd_ImStucc','Exterior2nd_Other','Exterior2nd_Stone','Exterior2nd_Stucco','MasVnrType_No','Foundation_CBlock','Foundation_Slab','Heating_GasA','Heating_OthW','Heating_Wall','Electrical_FuseF','Electrical_FuseP','Electrical_Mix','GarageType_BuiltIn','GarageType_CarPort','MiscFeature_Othr','MiscFeature_TenC','SaleType_Con','SaleType_ConLD','SaleType_ConLI','SaleType_ConLw','SaleType_New','SaleType_Oth','SaleCondition_AdjLand'], axis=1)


# In[ ]:





# In[100]:


#train = pd.concat([train_, Y], axis=1).reset_index(drop=True)   as: train_pyr2.csv


# # Fit Model

# In[64]:


X_train, X_test, y_train, y_test = train_test_split(train_, Y,test_size = 0.2)
print( X_train.shape, y_train.shape,X_test.shape, y_test.shape)


# In[57]:


from sklearn import metrics
from sklearn.model_selection import cross_val_score

def cross_val(model):
    pred = cross_val_score(model, train_, Y, cv=10)
    return pred.mean()

def print_evaluate(true, predicted):  
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    print('MAE:', mae)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('R2 Square', r2_square)
    
def evaluate(true, predicted):
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    return mae, mse, rmse, r2_square


# Linear Model

# In[65]:


## Linear Model
lr = LinearRegression()
lr.fit(X_train,y_train)
test_pre_lr = lr.predict(X_test)
train_pre_lr = lr.predict(X_train)


# In[66]:


cv = KFold(shuffle=True, random_state=2, n_splits=10)
scores = cross_val_score(lr, train_,Y,cv = cv, scoring = 'neg_mean_absolute_error')
print(scores)


# In[67]:


print('print the intercept :',lr.intercept_)


# In[68]:


coef_lr = pd.DataFrame(lr.coef_, train_.columns, columns=['Coefficient'])
coef_lr


# In[69]:


#plot between predicted values and residuals
plt.scatter(train_pre_lr, train_pre_lr - y_train, c = "blue",  label = "Training data")
plt.scatter(test_pre_lr,test_pre_lr - y_test, c = "black",  label = "Validation data")
plt.title("Linear regression")
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.legend(loc = "upper left")
plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")
plt.show()


# In[70]:


# Plot predictions - Real values
plt.scatter(train_pre_lr, y_train, c = "blue",  label = "Training data")
plt.scatter(test_pre_lr, y_test, c = "black",  label = "Validation data")
plt.title("Linear regression")
plt.xlabel("Predicted values")
plt.ylabel("Real values")
plt.legend(loc = "upper left")
plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")
plt.show()


# In[ ]:


#  As you can see, the regression line reduces much of the errors; therefore, performs much better than average line.


# In[71]:


print_evaluate(y_test, lr.predict(X_test))


# Ridge Regression

# In[72]:


## Importing Ridge ---> 1
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
ridge = Ridge()
parameters = {'alpha':[1e-15,1e-10,1e-8,1e-4,1e-3,1e-2,1,5,10,20]}
ridge_regressor = GridSearchCV(ridge, parameters, scoring ='neg_mean_squared_error',cv=5)
ridge_regressor.fit(X_train, y_train)


# In[73]:


ridge = RidgeCV(alphas = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60])
ridge.fit(X_train,y_train)
alpha = ridge.alpha_

iteration = 200
cv_ridge = KFold(shuffle=True, random_state=2, n_splits=10)
for i in range(iteration):
    ridge = RidgeCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, 
                          alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,
                          alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4],cv = cv_ridge)
    ridge.fit(X_train, y_train)
    alpha = ridge.alpha_
    train_pred_rdg = ridge.predict(X_train)
    test_pred_rdg = ridge.predict(X_test)
print("Best alpha :", alpha)


# In[74]:


scores_ridge = cross_val_score(ridge, train_,Y,cv = cv_ridge, scoring = 'neg_mean_absolute_error')
print(scores_ridge)


# In[75]:


coef_ridge = pd.Series(ridge.coef_, index = X_train.columns)

print("Ridge picked " + str(sum(coef_ridge != 0)) + " variables and eliminated the other " +  str(sum(coef_ridge == 0)) + " variables")
coef_ridge = pd.DataFrame(ridge.coef_, train_.columns, columns=['Coefficient'])
coef_ridge


# In[76]:


# Plot residuals
plt.scatter(train_pred_rdg, train_pred_rdg - y_train, c = "blue",  label = "Training data")
plt.scatter(test_pred_rdg, test_pred_rdg - y_test, c = "black", marker = "v", label = "Validation data")
plt.title("Linear regression with Ridge regularization")
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.legend(loc = "upper left")
plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")
plt.show()


# In[77]:


# Plot predictions - Real values
plt.scatter(train_pred_rdg, y_train, c = "blue",  label = "Training data")
plt.scatter(test_pred_rdg, y_test, c = "black",  label = "Validation data")
plt.title("Linear regression with Ridge regularization")
plt.xlabel("Predicted values")
plt.ylabel("Real values")
plt.legend(loc = "upper left")
plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")
plt.show()


# In[78]:


print_evaluate(y_test, ridge.predict(X_test))


# Lasso Regression

# In[79]:


lasso = LassoCV(alphas = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60])
lasso.fit(X_train,y_train)
alpha = lasso.alpha_
for i in range(iteration):
    cv_lasso = KFold(shuffle=True, random_state=2, n_splits=10)
    lasso = LassoCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, 
                          alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,
                          alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4],cv = cv_lasso)
    lasso.fit(X_train, y_train)
    alpha = lasso.alpha_
    train_pred_lasso = lasso.predict(X_train)
    test_pred_lasso = lasso.predict(X_test)
print("Best alpha :", alpha)


# In[80]:


scores_lasso = cross_val_score(lasso, train_,Y,cv = cv_lasso, scoring = 'neg_mean_absolute_error')
print(scores_lasso)


# In[81]:


coef_lasso = pd.Series(lasso.coef_, index = X_train.columns)

print("Lasso picked " + str(sum(coef_lasso != 0)) + " variables and eliminated the other " +  str(sum(coef_lasso == 0)) + " variables")
coef_lasso = pd.DataFrame(lasso.coef_, train_.columns, columns=['Coefficient'])
coef_lasso


# In[82]:


# Plot residuals
plt.scatter(train_pred_lasso, train_pred_lasso - y_train, c = "blue",  label = "Training data")
plt.scatter(test_pred_lasso , test_pred_lasso - y_test, c = "black", marker = "v", label = "Validation data")
plt.title("Linear regression with Lasso regularization")
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.legend(loc = "upper left")
plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")
plt.show()


# In[83]:


# Plot predictions - Real values
plt.scatter(train_pred_lasso, y_train, c = "blue",  label = "Training data")
plt.scatter(test_pred_lasso, y_test, c = "black",  label = "Validation data")
plt.title("Linear regression with Lasso regularization")
plt.xlabel("Predicted values")
plt.ylabel("Real values")
plt.legend(loc = "upper left")
plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")
plt.show()


# In[84]:


print_evaluate(y_test, lasso.predict(X_test))


# ElasticNet Regression

# In[85]:


elnet = ElasticNetCV(alphas = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60])
elnet.fit(X_train,y_train)
alpha = elnet.alpha_
for i in range(iteration):
    cv_elnet = KFold(shuffle=True, random_state=2, n_splits=10)
    elnet = ElasticNetCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, 
                          alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,
                          alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4],cv = cv_elnet)
    elnet.fit(X_train, y_train)
    alpha = elnet.alpha_
    train_pred_elnet = elnet.predict(X_train)
    test_pred_elnet = elnet.predict(X_test)
print("Best alpha :", alpha)


# In[86]:


scores_elnet = cross_val_score(elnet, train_,Y,cv = cv_elnet, scoring = 'neg_mean_absolute_error')
print(scores_elnet)


# In[87]:


coef_elnet = pd.Series(elnet.coef_, index = X_train.columns)

print("ElasticNet picked " + str(sum(coef_elnet != 0)) + " variables and eliminated the other " +  str(sum(coef_elnet == 0)) + " variables")
coef_elnet = pd.DataFrame(elnet.coef_, train_.columns, columns=['Coefficient'])
coef_elnet


# In[88]:


# Plot residuals
plt.scatter(train_pred_elnet, train_pred_elnet - y_train, c = "blue",  label = "Training data")
plt.scatter(test_pred_elnet, test_pred_elnet - y_test, c = "black", marker = "v", label = "Validation data")
plt.title("Linear regression with ElasticNet regularization")
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.legend(loc = "upper left")
plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")
plt.show()


# In[89]:


# Plot predictions - Real values
plt.scatter(train_pred_elnet, y_train, c = "blue",  label = "Training data")
plt.scatter(test_pred_elnet, y_test, c = "black",  label = "Validation data")
plt.title("Linear regression with ElasticNet regularization")
plt.xlabel("Predicted values")
plt.ylabel("Real values")
plt.legend(loc = "upper left")
plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")
plt.show()


# In[90]:


print_evaluate(y_test, elnet.predict(X_test))


# Compare Regression Results

# In[91]:


results_df = pd.DataFrame(data=[["Linear Regression", *evaluate(y_test, test_pre_lr) , cross_val(LinearRegression())]], 
                          columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])
results_df_2 = pd.DataFrame(data=[["Ridge Regression", *evaluate(y_test, test_pred_rdg) , cross_val(Ridge())]], 
                            columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])
results_df = results_df.append(results_df_2, ignore_index=True)
results_df_2 = pd.DataFrame(data=[["Lasso Regression", *evaluate(y_test, test_pred_lasso) , cross_val(Lasso())]], 
                            columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])
results_df = results_df.append(results_df_2, ignore_index=True)
results_df_2 = pd.DataFrame(data=[["ElasticNet Regression", *evaluate(y_test, test_pred_elnet) , cross_val(Lasso())]], 
                            columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])
results_df = results_df.append(results_df_2, ignore_index=True)
results_df


# In[92]:


Ridge_train_score = ridge.score(X_train,y_train)
Ridge_test_score = ridge.score(X_test, y_test)
print('Score for train ridge regressor:',Ridge_train_score)
print('Score for test ridge regressor',Ridge_test_score)
print('--------------------------------------------------')
Lasso_train_score = lasso.score(X_train,y_train)
Lasso_test_score = lasso.score(X_test, y_test)
print('Score for train lasso regressor:',Lasso_train_score)
print('Score for test lasso regressor',Lasso_test_score)
print('--------------------------------------------------')
elnet_train_score = elnet.score(X_train,y_train)
elnet_test_score = elnet.score(X_test, y_test)
print('Score for train ElasticNet regressor:',elnet_train_score)
print('Score for test ElasticNet regressor',elnet_test_score)


# In[ ]:





# In[93]:


# Output feature importance coefficients, map them to their feature name, and sort values
coef = pd.Series(ridge.coef_ , index = train_.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 5))
coef.head(25).plot(kind='bar')
plt.title('Feature Significance')
plt.tight_layout()


# In[ ]:


## 1)Output feature importance coefficients ; 2)Map coefficients to their feature name ; 3)Sort features in descending order


# In[94]:


# Output feature importance coefficients, map them to their feature name, and sort values
ridge.fit(train_, Y) 
coef = pd.Series(ridge.coef_ , index = train_.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 5))
coef.head(25).plot(kind='bar', color= 'red')
plt.title('Feature Significance')
plt.tight_layout()


# In[95]:


from sklearn import tree
from sklearn.tree import DecisionTreeRegressor

## decision tree
dectree = tree.DecisionTreeRegressor(max_depth=5)
dectree.fit(X_train,y_train)
dectree.score(X_test,y_test)


# In[96]:


y_train_dectree = dectree.predict(X_train)
y_test_dectree = dectree.predict(X_test)


# In[97]:


# Plot the results
plt.figure()
plt.plot(X_train, y_train_dectree, color="cornflowerblue", linewidth=2)
plt.plot(X_test, y_test_dectree, color="yellowgreen", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()


# In[98]:


from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor(max_depth=2, random_state=0)
regr.fit(X_train, y_train)
regr_pred_train= regr.predict(X_train)
regr_pred_test= regr.predict(X_test)
regr.score(X_test,y_test)


# In[99]:


regr.get_params(deep=True)


# In[100]:


print('Score for train RandomForest regressor:',regr.score(X_train, y_train))
print('Score for test RandomForest regressor',regr.score(X_test, y_test))


# In[ ]:





# In[ ]:





# In[101]:


importances = regr.feature_importances_
indices = np.argsort(importances)[::-1]
print("Feature ranking:")

for f in range(train_.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))


# In[102]:


from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
rng = np.random.RandomState(1)

# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=4)

regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                          n_estimators=300, random_state=rng)

regr_1.fit(X_train, y_train)
regr_2.fit(X_train, y_train)

# Predict
train_pred_regr1 = regr_1.predict(X_train)
test_pred_regr1 = regr_1.predict(X_test)
train_pred_regr2 = regr_2.predict(X_train)
test_pred_regr2 = regr_2.predict(X_test)


# Plot the results
plt.figure()
#plt.scatter(X_train, y_train , c="k", label="training samples")
plt.plot(X_train, train_pred_regr1, c="g", linewidth=2)
plt.plot(X_train, train_pred_regr2, c="r", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Boosted Decision Tree Regression")
plt.legend()
plt.show()


# In[141]:


ridge_coef = pd.Series(ridge.coef_ , index = train_.columns)
ridge_coef.to_csv("ridgecoef.csv")


# In[142]:


lasso_coef = pd.Series(lasso.coef_ , index = train_.columns)
lasso_coef.to_csv("lassocoef.csv")


# In[143]:


elnet_coef = pd.Series(elnet.coef_ , index = train_.columns)
elnet_coef.to_csv("elnetcoef.csv")


# In[103]:


Test_lr = lr.predict(test)
Test_rdg = ridge.predict(test)
Test_lasso = lasso.predict(test)
Test_elnet = elnet.predict(test)
Test_dectree = dectree.predict(test)
Test_regr_1 = regr_1.predict(test)
Test_regr_2 = regr_2.predict(test)


# In[104]:


Test_rdge_price =Test_rdg


# In[146]:


import pickle


# In[154]:


file_name= "lr.pkl"
with open(file_name, 'wb') as file:
    pickle.dump(lr, file)
    
file_name= "ridge.pkl"
with open(file_name, 'wb') as file:
    pickle.dump(ridge, file)
    
file_name= "lasso.pkl"
with open(file_name, 'wb') as file:
    pickle.dump(lasso, file)
    
file_name= "elnet.pkl"
with open(file_name, 'wb') as file:
    pickle.dump(elnet, file)
    
file_name= "dectree.pkl"
with open(file_name, 'wb') as file:
    pickle.dump(dectree, file)
    
file_name= "regr_1.pkl"
with open(file_name, 'wb') as file:
    pickle.dump(regr_1, file)
    
file_name= "regr_2.pkl"
with open(file_name, 'wb') as file:
    pickle.dump(regr_2, file)


# In[161]:


PricePrediction = pd.DataFrame(Test_rdge_price)
test = pd.DataFrame(test)
test_data = pd.concat([test, Test_rdge_price], axis=1).reset_index(drop = True)


# In[162]:


#Test.to_csv("test_data.csv")


# In[155]:


object_ridge = pd.read_pickle(r'C:\\Users\\lenovo\ridge.pkl')


# In[157]:


Y_test_price = object_ridge.predict(test)


# In[159]:


Test_rdg


# In[160]:


Y_test_price


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




