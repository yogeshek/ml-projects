import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

#project path
PROJECT_ROOT =Path(__file__).parent.parent.parent.parent
DATA_PATH = PROJECT_ROOT /'datasets'/'raw'/'AmesHousing.xls'
MODELS_PATH = PROJECT_ROOT / 'models'

#load data
df =pd.read_excel(DATA_PATH, engine='xlrd')
# data = df.head(n=5)
# print(data)

print(f"data shape: {df.shape}")
print(df.head)


# df = df.select_dtypes(include=['int32','int64']).copy()
# df.head()
# print(f"number of int columne: {len(int_col)}")
# print(f"total columns: {len(df.columns)}")
# print(df.loc[:5, int_col[:]])
# df_clean=df[int_col]

##################################################################################################
# # print(f"\nData type: \n{df.dtypes.value_counts()}")
# print(f"\nMissing values: {df.isnull().sum().sum()} total")
# print(f"\nTarget vriable range: {df['SalePrice'].min():.0f} to {df['SalePrice'].max():.0f}")

# #***********OUTLIERS HANDLING*******************

#5 number summary
min_val = df['SalePrice'].min()
Q1 = df['SalePrice'].quantile(0.25)
median = df['SalePrice'].median()
Q3 = df['SalePrice'].quantile(0.75)
max_val = df['SalePrice'].max()

IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"lower bound: {lower_bound}")
sales_outliers = df[(df['SalePrice'] < lower_bound) | (df['SalePrice'] > upper_bound)]
# print(sales_outliers)
print(f"outliers row counts: {len(sales_outliers)}")
df_clean = df[~((df['SalePrice'] < lower_bound) | (df['SalePrice'] > upper_bound))]



# print(f"row count: {len(df)}")
# print(f"clean row count: {len(df_clean)}")

# #**********NULL VALUE HANDLING******************

# null_counts = df_clean.isnull().sum()
# null_percentage = (df_clean.isnull().sum() / len(df_clean)) * 100
# coulmns_with_nulls = null_percentage[null_percentage>0].sort_values(ascending=False)
# print(f"number of null: {coulmns_with_nulls}")
# print(null_counts)

# # df_clean = df_clean.dropna() # any row with null value dropped
# # print(f"row count after dropping null: {len(df_clean)}")

# #1 drop column with high null percentage

# print(len(df_clean.columns))
# threshhold = 50
# col_to_drop = null_percentage[null_percentage> threshhold].index
# print(f"********************column with moretahn 50% null: {col_to_drop}")
# df_clean = df_clean.drop(columns = col_to_drop)
# print(len(df_clean.columns))

# #2 drop rows(if few nulls)
# #only if very few rows have nulls
# df_clean = df_clean.dropna()
# #drop rows where specific column is null
# df_clean = df_clean.dropna(subset=['SalePrice'])


# #3. fill with appropriate values based on data type
# #for nymeric columns
# numeric_columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()
# print(f"Numerical columns: {numeric_columns}")

# df_clean['Order']=df_clean['Order'].fillna(df_clean['Order'].mean())
# df_clean['Lot Frontage']=df_clean['Lot Frontage'].fillna(df_clean['Lot Frontage'].mean())
# df_clean['Lot Area']=df_clean['Lot Area'].fillna(df_clean['Lot Area'].mean())
# df_clean['Lot Frontage']=df_clean['Lot Frontage'].fillna(df_clean['Lot Frontage'].mean())

# #for categorical columns

# categorical_columns = df_clean.select_dtypes(include=['object']).columns.tolist()
# print(f"Categorical columns: {categorical_columns}")

# df_clean['Street']=df_clean['Street'].fillna(df['Street'].mode()[0])
# df_clean['Lot Shape']=df_clean['Lot Shape'].fillna('unknown') # custom fill
# df_clean['Lot Config']=df_clean['Lot Config'].fillna(method='ffill') # forward fill
# df_clean['Lot Area']=df_clean['Lot Area'].fillna(method='bfill') # backward fill

# #4. Advance : use algorithms (KNN, iterative imputer)

# from sklearn.impute import SimpleImputer, KNNImputer
# imputer = SimpleImputer(strategy='mean')
# df_clean['Overall Qual']=imputer.fit_transform(df_clean[['Overall Qual']])

# knn_imputer = KNNImputer(n_neighbors=5)
# df_clean['Overall Qual']=knn_imputer.fit_transform(df_clean[['Overall Qual']])

# print(df.shape)
# print(df_clean.shape)
# print(df_clean.head)
# print(df_clean.info)

#PREPARE FEATURE AND TARGET

x = df_clean.drop('SalePrice', axis=1)
y = df_clean['SalePrice']
# x = pd.get_dummies(x, drop_first=True)
#create binary(0/1) columns for each unique category in each categorical column
#drop_first = True remove the first categorical vatiable to avoid multicoleniarity

#split data (80:20)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size= 0.2,
    random_state= 42
)

#data preprocessing

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


total_columns = x_train.columns.tolist()
categorical_cols = x_train.select_dtypes(include=['object','category']).columns.tolist()
numeric_cols = x_train.select_dtypes(include=['int64','float64']).columns.tolist()
print(f"total columns: {len(total_columns)}")
print(f"Categorical columns: {len(categorical_cols)}")
print(f"Numeric columns: {len(numeric_cols)}")

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer',SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]),numeric_cols),
        ('cat', Pipeline(steps=[
            ('imputer',SimpleImputer(strategy='constant',fill_value='MISSING')),
            ('onehot',OneHotEncoder(handle_unknown='ignore', drop='first'))
        ]),categorical_cols)
    ]
)

from sklearn.ensemble import RandomForestRegressor
model = Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('regressor',RandomForestRegressor(random_state=42, n_jobs=-1))
])

# #scale features
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# x_train_scaled = scaler.fit_transform(x_train)
# x_test_scaled = scaler.transform(x_test)

# #model
# from sklearn.linear_model import LinearRegression
# model = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('regressor', LinearRegression())
# ])
model.fit(x_train,y_train)

#prediction
y_pred = model.predict(x_test)

#evaluation of the model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
mae = mean_absolute_error(y_test,y_pred)

print(f"r2_score = : {r2:.4f}")
print(f"root mean square error: {rmse:.2f}")
print(f"mean absolute error: {mae:.2f}")













