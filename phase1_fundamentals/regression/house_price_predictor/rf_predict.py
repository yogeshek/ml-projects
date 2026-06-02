#1 Import and setup
import pickle
import numpy as np
from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
MODELS_PATH = PROJECT_ROOT / 'models'

#step2 load model function

def load_models():
    with open(MODELS_PATH / 'random_forest_housing.pkl', 'rb') as f:
        model = pickle.load(f)
    # with open(MODELS_PATH / 'linear_scalar_housing.pkl', 'rb') as f:
    #     scaler = pickle.load(f)
    return model #, scaler

#step3 predict function

def predict_price(features):
    """
    Predict house price given the features
    
    Args:
        features: dict or array-like with 8 features in order
                    MedInc, HouseAge, Averooms, AveBedrms, Population, AveOccup, Lattitude, Longitude
                    
    returns:
        Predicted price: (in $100000s)
    """
    
    # model, scaler = load_models()
    model = load_models()
    
    #convert dict to arry if needed
    
    # Convert dict to array if needed
    if isinstance(features, dict):
        #if the features are given as dictonary[feature:value]
        feature_order = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
                        'Population', 'AveOccup', 'Latitude', 'Longitude']
        #feature is order/picked as it defined in the feature_order
        features_array = np.array([[features[k] for k in feature_order]])
    else:
        #if only the values are given
        #reshape(1,-1) - reshape to 2D, without it 8 sample with 1 feature, but ml expect 2d, 1 sample with 8 features. 
        features_array = np.array(features).reshape(1, -1)
        
    features_df = pd.DataFrame(features_array, columns=feature_order)
        
    #scale features
    # features_scaled = scaler.transform(features_df)
    #scaled_value = (original_value - mean) / standard_deviation, mean and sd from training
    #transform Use the means and stds calculated during .fit()
    
    #prediction
    prediction = model.predict(features_df)
    #model learned co-efficient during the training
    #coefficient = sum((size_i - mean_size) × (price_i - mean_price)) / sum((size_i - mean_size)²)
    #eg- coeff = [0.5, 0.3, -0.2, 0.1, 0.4, -0.1, 0.8, -0.6]
    # intercept = eg 100000
    #prediction = (0.5*0.85(scaled feature1 of test) + 0.3*0.32(scaled feature2 of test) + -0.2*-0.15 + ...) + 100000
    #or
    #[b, c1, c2] × X_with_intercept = y , add 1 column as intercept and this formula will give b, c1 and c2 directly
    
    return prediction[0]
#4 main execution

if __name__=="__main__":
    example_house = {
        'MedInc': 3.5,        # Median income in block group (in $10,000s)
        'HouseAge': 25.0,     # Median house age in block group
        'AveRooms': 5.5,      # Average number of rooms per household
        'AveBedrms': 1.2,     # Average number of bedrooms per household
        'Population': 1500.0, # Block group population
        'AveOccup': 3.0,      # Average number of household members
        'Latitude': 34.05,    # Block group latitude
        'Longitude': -118.25  # Block group longitude
    }
    
    try:
        predicted_price = predict_price(example_house)
        print(f"\nPredcited House Price: {predicted_price*100000:,.2f}")
    except FileNotFoundError:
        print("Error- model file not found - run train.py")
    except Exception as e:
        print(f"exception during prediction: {e}")
    
    
    
        
    
    