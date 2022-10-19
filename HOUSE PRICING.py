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
df_train = df_train.dropna()
x_train = df_train.drop(['SalePrice'], axis=1).values
y_train = df_train['SalePrice'].values
df_test = df_test.drop(['Id'], axis=1)
x_test = df_test.values

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
scaler.fit(x_test)
x_test = scaler.transform(x_test)
#-----------------------------linear regression----------------------------------------------
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as MSE, r2_score as ACC

results = []
model_reg = LinearRegression()
score = cross_validate(model_reg,x_train,y_train, cv=5, scoring=['neg_mean_squared_error','neg_root_mean_squared_error' , 'r2'] )
results.append({'model': 'linear regression' , 'MSE':np.mean(score['test_neg_mean_squared_error']) , 'RMSE':np.mean(score['test_neg_root_mean_squared_error']) , 'ACC': np.mean(score['test_r2'])})

#-----------------------sgd------------------------------------------
from sklearn.linear_model import SGDRegressor
model_sgd = SGDRegressor()
score = cross_validate(model_sgd, x_train , y_train, cv=5 , scoring=['neg_mean_squared_error','neg_root_mean_squared_error' , 'r2'])
results.append({'model': 'SGD' , 'MSE':np.mean(score['test_neg_mean_squared_error']) , 'RMSE':np.mean(score['test_neg_root_mean_squared_error']) , 'ACC': np.mean(score['test_r2'])})
#------------------polynomial----------------------------------------
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV

parameters = {'polynomial_features__degree':[1,2,3]}
pipline = Pipeline([
                  ('polynomial_features', PolynomialFeatures()),
                  ('linear_regressor', SGDRegressor())
                ])

model = GridSearchCV(pipline, parameters)
score = cross_validate(model , x_train, y_train , cv = 5 , scoring = ['neg_mean_squared_error' , 'neg_root_mean_squared_error' , 'r2'])
results.append({'Model':'Poly Regression' , 'MSE':np.mean(score['test_neg_mean_squared_error']) , 'RMSE':np.mean(score['test_neg_root_mean_squared_error']) , 'ACC': np.mean(score['test_r2'])})
#-----------------------------SVR--------------------------------------------------
from sklearn.svm import SVR

model = SVR()
score = cross_validate(model , x_train , y_train , cv = 5 , scoring = ['neg_mean_squared_error' ,'neg_root_mean_squared_error' , 'r2'])
results.append({'Model':'SVM Regression' , 'MSE':np.mean(score['test_neg_mean_squared_error']) , 'RMSE':np.mean(score['test_neg_root_mean_squared_error']) , 'ACC': np.mean(score['test_r2'])})
print(results)
#----------------comparing the models------------------------------------------
for row in results:
  if row['ACC'] < 0 :
    row['ACC'] = 0

fig , ax = plt.subplots(nrows = 3 , ncols =1)
fig.set_figheight(15)
fig.set_figwidth(8)

pos = [row['model'] for row in results]

acc = [row['ACC'] for row in results]
ax[0].bar(pos , acc)
ax[0].set_title('ACC')

mse = [row['MSE'] for row in results]
ax[1].bar(pos , mse)
ax[1].set_title('MSE')

rmse = [row['RMSE'] for row in results]
ax[2].bar(pos , rmse)
ax[2].set_title('RMSE')

plt.show()
#-----------------learning curve on best model--------------------------
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
def curve(model, x, y):
    x_train , x_val , y_train, y_val = train_test_split(x , y , test_size=0.4)
    train_error = []
    val_error = []
    for m in range (1 , len(x_train) , 10):
        model.fit(x_train[:m] , y_train[:m])
        y_trian_pred = model.predict(x_train[:m])
        y_val_pred = model.predict(x_val)

        train_error.append(mean_squared_error(y_train[:m] ,y_trian_pred))
        val_error.append(mean_squared_error(y_val , y_val_pred))
    return train_error , val_error

model = Pipeline([
                  ('linear_regressor', SGDRegressor())
                ])

train_errors , val_errors = curve(model , x_train , y_train)

plt.plot(train_errors , 'r-' , label = 'train error')
plt.plot(val_errors , 'b-' , label = 'val error')
plt.ylabel('MSE')
plt.legend()
plt.show()
