import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Create sample data for testing (replace with your actual data loading)
np.random.seed(42)
df = pd.DataFrame({
    'V1': [1.5, np.nan, 3.2, 4.1, 5.3, 2.1, np.nan, 3.8, 4.5, 6.2],
    'V2': [2.3, 3.1, np.nan, 5.2, 6.1, 3.5, 4.2, np.nan, 5.8, 7.1],
    'V3': [10, 20, 30, 40, 50, 15, 25, 35, 45, 55],
    'V4': [100, 200, 300, 400, 500, 150, 250, 350, 450, 550],
    'V5': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C'],
    'Target': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No']
})

# Load and prepare data
df_copy = df.copy()
df_copy = df_copy.replace('?', np.nan)

X = df_copy.drop('Target', axis=1)
y = df_copy['Target']

# Encode target variable using OrdinalEncoder as specified
oe_target = OrdinalEncoder()
y_encoded = oe_target.fit_transform(y.values.reshape(-1, 1)).ravel()

print("Original data shape:", X.shape)
print("Features:", X.columns.tolist())

# Build preprocessing pipeline according to the diagram
# Branch 1: SimpleImputer -> StandardScaler for Features 1,2,3,4 (V1, V2, V3, V4)
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # Imputes V1, V2 (FeatureIndex=[0,1])
    ('scaler', StandardScaler())  # Scales V1, V2, V3, V4 (FeatureIndex=[0,1,2,3])
])

# Branch 2: OrdinalEncoder for Feature 5 (V5)
categorical_pipeline = Pipeline([
    ('encoder', OrdinalEncoder())  # Encodes V5 (FeatureIndex=[4])
])

# FeatureUnion: Combine both branches
preprocessor = ColumnTransformer([
    ('numeric', numeric_pipeline, ['V1', 'V2', 'V3', 'V4']),  # Features 1-4
    ('categorical', categorical_pipeline, ['V5'])  # Feature 5
])

# Apply preprocessing
X_preprocessed = preprocessor.fit_transform(X)

print("\nPreprocessed data shape:", X_preprocessed.shape)

# RFE with LogisticRegression to find 2 most important features
lr = LogisticRegression(random_state=42, max_iter=1000)
rfe = RFE(estimator=lr, n_features_to_select=2)
X_rfe = rfe.fit_transform(X_preprocessed, y_encoded)

# Get selected feature names
feature_names = ['V1', 'V2', 'V3', 'V4', 'V5']
selected_features = [feature_names[i] for i in range(len(feature_names)) if rfe.support_[i]]
feature_ranking = rfe.ranking_

print("\n" + "="*50)
print("RFE RESULTS with LogisticRegression")
print("="*50)
print(f"Selected features: {selected_features}")
print(f"Feature ranking: {dict(zip(feature_names, feature_ranking))}")
print(f"Support mask: {rfe.support_}")
print(f"\nThe two most important features computed by RFE are: {', '.join(selected_features)}")