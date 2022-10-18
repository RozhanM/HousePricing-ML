import numpy as np
import matplotlib as mlt
import seaborn as sns
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
#print(df_train.head())
#print(df_train.isna().any())

col_name = []
null_num = []
for column in df_train :
    per = df_train[column].isna().mean() * 100
    if per :
        col_name.append(column)
        null_num.append(per)
nan_df = pd.DataFrame({'columns' : col_name , 'percent' : null_num})
#print(nan_df.head())

drop_cols = nan_df[nan_df['percent'] >= 80]['columns']
df_train = df_train.drop(drop_cols, axis=1)
df_test = df_test.drop(drop_cols, axis= 1)
#print(df_train.head())

cat_cols = df_train.select_dtypes(include=['object'])
num_cols = df_train.select_dtypes(exclude=['object'])
cat_nulls = cat_cols.isnull().sum()
num_nulls = num_cols.isnull().sum()
#print(cat_nulls)
#print(num_nulls)

#-------------using KNN for numerical columns------------
X = df_train['LotFrontage'].values
X = X.reshape(len(X), 1)
imputer = KNNImputer()
imputer.fit(X)
LotFrontage_trans = imputer.transform(X)
df_train['LotFrontage'] = LotFrontage_trans

X = df_train['GarageYrBlt'].values
X = X.reshape(len(X), 1)
imputer = KNNImputer()
imputer.fit(X)
GarageYrBlt_trans = imputer.transform(X)
df_train['GarageYrBlt'] = GarageYrBlt_trans

X = df_test['LotFrontage'].values
X = X.reshape(len(X), 1)
imputer = KNNImputer()
imputer.fit(X)
LotFrontage_trans = imputer.transform(X)
df_test['LotFrontage'] = LotFrontage_trans

X = df_test['GarageYrBlt'].values
X = X.reshape(len(X), 1)
imputer = KNNImputer()
imputer.fit(X)
GarageYrBlt_trans = imputer.transform(X)
df_test['GarageYrBlt'] = GarageYrBlt_trans

num_cols2 = df_train.select_dtypes(exclude=['object'])
num_nulls2 = num_cols2.isnull().sum()
#print(num_nulls2)

cat_cols = cat_cols.fillna(method='bfill')
cat_cols = df_test.select_dtypes(include=['object'])
cat_cols = cat_cols.fillna(method='bfill')
#-------------------------------------------------------------------
numerical_columns = df_train.select_dtypes(exclude='object')
numerical_columns = numerical_columns.astype('float')
columns = numerical_columns.columns
for column in columns :
    df_train[column] = numerical_columns[column]

numerical_columns = df_test.select_dtypes(exclude='object')
numerical_columns = numerical_columns.astype('float')
columns = numerical_columns.columns
for column in columns :
    df_test[column] = numerical_columns[column]

categorical_columns = df_train.select_dtypes(include='object')
categorical_columns = categorical_columns.astype('category')

categorical_columns[categorical_columns.columns] = categorical_columns[categorical_columns.columns].apply(lambda x : x.cat.codes)
for column in categorical_columns.columns :
    df_train[column] = categorical_columns[column]

categorical_columns = df_test.select_dtypes(include='object')
categorical_columns = categorical_columns.astype('category')
categorical_columns[categorical_columns.columns] = categorical_columns[categorical_columns.columns].apply(lambda x : x.cat.codes)
for column in categorical_columns.columns :
    df_test[column] = categorical_columns[column]

df_train = df_train.drop(['Id'], axis= 1)
x_train = df_train.drop(['SalePrice'], axis=1).values
y_train = df_train['SalePrice'].values
df_test = df_test.drop(['Id'], axis=1)
x_test = df_test.values

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
scaler.fit(x_test)
x_test = scaler.transform(x_test)
#---------------------------------------------------------------------------