#import and setup
import pickle
import numpy as np
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
MODEL_PATH = PROJECT_ROOT / 'models'
DATA_PATH = PROJECT_ROOT / 'datasets' / 'raw' / 'AmesHousing.xls'

from train import remove_outliers
def load_clean_data():
    df = pd.read_excel(DATA_PATH, engine='xlrd')
    df_clean = remove_outliers(df, column='SalePrice')
    #     # Remove outliers (same logic as train.py)
    # Q1 = df['SalePrice'].quantile(0.25)
    # Q3 = df['SalePrice'].quantile(0.75)
    # IQR = Q3 - Q1
    # lower_bound = Q1 - 1.5 * IQR
    # upper_bound = Q3 + 1.5 * IQR
    # df_clean = df[~((df['SalePrice'] < lower_bound) | (df['SalePrice'] > upper_bound))]
    return df_clean

#2 load the models

def load_models():
    with open(MODEL_PATH / 'rf_linear_regression_housing.pkl','rb') as f:
        model = pickle.load(f)
    return model
        
def predict_price(features):
    model = load_models()
    features_df = pd.DataFrame([features])
    prediction = model.predict(features_df)
    return prediction[0]
    

if __name__=="__main__":
    # example_house = {
    #     'Order': 1,        # Median income in block group (in $10,000s)
    #     'PID': 526350040,     # Median house age in block group
    #     'MS SubClass': 141.0,      # Average number of rooms per household
    #     'MS Zoning': 31770,     # Average number of bedrooms per 
    #     'Lot Frontage': 0.1,
    # }
    
    df_clean = load_clean_data()
    example_house = df_clean.drop('SalePrice', axis=1).iloc[17].to_dict()

    
    try:
        predicted_price = predict_price(example_house)
        print(f"\nPredicted House Price: ${predicted_price:,.2f}")
    except FileNotFoundError:
        print("Error- model file not found - run train.py")
    except Exception as e:
        print(f"exception during prediction: {e}")
    
