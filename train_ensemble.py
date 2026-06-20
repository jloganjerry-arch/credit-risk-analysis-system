import pandas as pd
import numpy as np
import pickle
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

print("Loading dataset...")
df = pd.read_csv("application_data.csv", usecols=[
    'TARGET', 'NAME_CONTRACT_TYPE', 'DAYS_BIRTH', 'AMT_CREDIT', 
    'AMT_INCOME_TOTAL', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 
    'OCCUPATION_TYPE', 'DAYS_EMPLOYED'
])

print("Preprocessing...")
df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)

features = [
    'NAME_CONTRACT_TYPE',
    'DAYS_BIRTH',
    'AMT_CREDIT',
    'AMT_INCOME_TOTAL',
    'NAME_INCOME_TYPE',
    'NAME_EDUCATION_TYPE',
    'OCCUPATION_TYPE',
    'DAYS_EMPLOYED'
]

X = df[features]
y = df['TARGET']

print("Handling missing values...")
imputer = SimpleImputer(strategy='most_frequent')
X = pd.DataFrame(imputer.fit_transform(X), columns=features)

print("Feature Engineering...")
X['DAYS_BIRTH'] = abs(X['DAYS_BIRTH'].astype(float)) / 365
X['DAYS_EMPLOYED'] = abs(X['DAYS_EMPLOYED'].astype(float))

print("Encoding categorical features...")
categorical_columns = [
    'NAME_CONTRACT_TYPE',
    'NAME_INCOME_TYPE',
    'NAME_EDUCATION_TYPE',
    'OCCUPATION_TYPE'
]

label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

for col in X.columns:
    X[col] = X[col].astype(float)

# Determine class imbalance ratio for scale_pos_weight
neg_class = (y == 0).sum()
pos_class = (y == 1).sum()
imbalance_ratio = neg_class / pos_class
print(f"Class imbalance ratio: {imbalance_ratio:.2f}")

print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# We train the ensemble models on train set
# 1. CatBoost
print("Training CatBoost model...")
cat_model = CatBoostClassifier(
    iterations=200, 
    depth=6, 
    learning_rate=0.1, 
    random_state=42, 
    verbose=0, 
    scale_pos_weight=imbalance_ratio
)
cat_model.fit(X_train, y_train)

# 2. LightGBM
print("Training LightGBM model...")
lgbm_model = LGBMClassifier(
    n_estimators=200, 
    max_depth=6, 
    learning_rate=0.1, 
    random_state=42, 
    scale_pos_weight=imbalance_ratio, 
    verbose=-1,
    n_jobs=-1
)
lgbm_model.fit(X_train, y_train)

# 3. XGBoost
print("Training XGBoost model...")
xgb_model = XGBClassifier(
    n_estimators=200, 
    max_depth=6, 
    learning_rate=0.1, 
    random_state=42, 
    scale_pos_weight=imbalance_ratio,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)

print("Saving models...")
with open("model_catboost.pkl", "wb") as f:
    pickle.dump(cat_model, f)

with open("model_lightgbm.pkl", "wb") as f:
    pickle.dump(lgbm_model, f)

with open("model_xgboost.pkl", "wb") as f:
    pickle.dump(xgb_model, f)

print("All ensemble models trained and saved successfully!")
