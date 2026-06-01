#1 Import libraries

import pandas as  pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
from pathlib import Path
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

#2. setup path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
# DATA_PATH = PROJECT_ROOT / 'data' / 'your_data.csv'
MODELS_PATH = PROJECT_ROOT / 'models'

#3. Load data

housing = fetch_california_housing(as_frame=True)
df = housing.frame
print(df.head())
print(df.info())

#4. prepare feature and target

x = df[housing.feature_names] # Features 8 columns
y = df['MedHouseVal'] # Target (house price)

#5. split data (80:20)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42
)

#6 Scale features

scaler = StandardScaler()
#store the parameters learned from the training data.
x_train_scaled = scaler.fit_transform(x_train)
#fit() = eg- teacher calculates class average and std
#transform() = teacher converts marks to z-scores - =x−μ​/σ
#fit_transform() = both together in one call
x_test_scaled = scaler.transform(x_test)
#useses the calculated value from trianing data

#7 create and train model

model = LinearRegression()
# y=w1​x1​+w2​x2​+⋯+wn​xn​+b
model.fit(x_train_scaled, y_train)
#This is where the actual mathematics happens.
#model learns coefficient and intercept, It chooses w and b so that prediction errors are minimized
#MSE=n1​∑i=1n​(yi​−y^​i​)2
#Eg- find the best line y=3x+2 (co efficient 3 and intercept 2)


#8 make prediction

y_pred = model.predict(x_test_scaled)
#predict the y valude for the test data using best line- y=3x+2

#9 Evaluate model

r2 = r2_score(y_test, y_pred) # co efficient determination - how much of price variation explained
rmse = np.sqrt(mean_squared_error(y_test, y_pred)) # the typical error the model makes - "Typically, my predictions are off by about eg $25,000"
mae = mean_absolute_error(y_test, y_pred) # the average absolute error - "On average, my predictions are eg $18,000 away from actual prices"

print(f"R2 Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")

#10 save model and scalar

MODELS_PATH.mkdir(exist_ok=True)

with open(MODELS_PATH / 'linear_regression_housing.pkl', 'wb') as f:
    pickle.dump(model, f)
with open(MODELS_PATH / 'linear_scalar_housing.pkl', 'wb') as f:
    pickle.dump(scaler, f)
    



