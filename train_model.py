# =========================================================
# ADABOOST CREDIT FRAUD DETECTION MODEL
# FINAL TRAIN_MODEL.PY
# =========================================================

# =========================================================
# IMPORT LIBRARIES
# =========================================================

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import LabelEncoder


# =========================================================
# LOAD DATASET
# =========================================================

print("Loading dataset...")

df = pd.read_csv(

    "application_data.csv"

)


# =========================================================
# SELECT FEATURES
# =========================================================

features = [

    'NAME_CONTRACT_TYPE',
    'DAYS_BIRTH',
    'AMT_CREDIT',
    'AMT_INCOME_TOTAL',
    'NAME_INCOME_TYPE',
    'NAME_EDUCATION_TYPE',
    'OCCUPATION_TYPE',
    'DAYS_EMPLOYED',

    
]


# =========================================================
# INPUT FEATURES
# =========================================================

X = df[features]

y = df['TARGET']


# =========================================================
# CLEAN DAYS EMPLOYED
# =========================================================

print("Cleaning DAYS_EMPLOYED...")

X['DAYS_EMPLOYED'] = X['DAYS_EMPLOYED'].replace(

    365243,
    np.nan

)


# =========================================================
# HANDLE MISSING VALUES
# =========================================================

print("Handling missing values...")

imputer = SimpleImputer(

    strategy='most_frequent'

)

X = pd.DataFrame(

    imputer.fit_transform(X),

    columns=features

)


# =========================================================
# CONVERT AGE
# =========================================================

print("Converting age...")

X['DAYS_BIRTH'] = abs(

    X['DAYS_BIRTH']

) // 365


# =========================================================
# CONVERT DAYS EMPLOYED
# =========================================================

X['DAYS_EMPLOYED'] = abs(

    X['DAYS_EMPLOYED']

)


# =========================================================
# FEATURE ENGINEERING
# =========================================================

print("Creating advanced features...")


# ---------------------------------------------------------
# CREDIT / INCOME RATIO
# ---------------------------------------------------------




# ---------------------------------------------------------
# EMPLOYMENT STABILITY RATIO
# ---------------------------------------------------------




# =========================================================
# LABEL ENCODING
# =========================================================

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

    X[col] = le.fit_transform(

        X[col].astype(str)

    )

    label_encoders[col] = le


# =========================================================
# TRAIN TEST SPLIT
# =========================================================

print("Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42,

    stratify=y

)


# =========================================================
# CREATE ADABOOST MODEL
# =========================================================

print("Creating AdaBoost model...")


base_model = DecisionTreeClassifier(

    max_depth=3,

    random_state=42

)


model = AdaBoostClassifier(

    estimator=base_model,

    n_estimators=300,

    learning_rate=0.05,

    random_state=42

)


# =========================================================
# TRAIN MODEL
# =========================================================

print("Training model...")

model.fit(

    X_train,
    y_train

)


# =========================================================
# PREDICT PROBABILITIES
# =========================================================

print("Making predictions...")

probs = model.predict_proba(X_test)[:, 1]


# =========================================================
# THRESHOLD TUNING
# =========================================================

threshold = 0.70

y_pred = (probs > threshold).astype(int)


# =========================================================
# ACCURACY
# =========================================================

accuracy = accuracy_score(

    y_test,
    y_pred

)

print("\n===================================")
print(f"Accuracy: {accuracy * 100:.2f}%")
print("===================================")


# =========================================================
# CLASSIFICATION REPORT
# =========================================================

print("\nClassification Report:\n")

print(

    classification_report(

        y_test,
        y_pred

    )

)


# =========================================================
# CONFUSION MATRIX
# =========================================================

print("\nConfusion Matrix:\n")

print(

    confusion_matrix(

        y_test,
        y_pred

    )

)


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

print("\n===================================")
print("FEATURE IMPORTANCE")
print("===================================")

importance = model.feature_importances_


for feature, score in zip(X.columns, importance):

    print(f"{feature} --> {round(score, 4)}")


# =========================================================
# SAVE MODEL
# =========================================================

print("\nSaving model...")

with open(

    "credit_fraud_model.pkl",
    "wb"

) as f:

    pickle.dump(

        model,
        f

    )


# =========================================================
# SAVE LABEL ENCODERS
# =========================================================

print("Saving label encoders...")

with open(

    "label_encoders.pkl",
    "wb"

) as f:

    pickle.dump(

        label_encoders,
        f

    )


# =========================================================
# SAVE IMPUTER
# =========================================================

print("Saving imputer...")

with open(

    "imputer.pkl",
    "wb"

) as f:

    pickle.dump(

        imputer,
        f

    )


# =========================================================
# MODEL SAVED
# =========================================================

print("\n===================================")
print("MODEL TRAINING COMPLETED")
print("FILES SAVED SUCCESSFULLY")
print("===================================")

print("\nSaved Files:")
print("1. credit_fraud_model.pkl")
print("2. label_encoders.pkl")
print("3. imputer.pkl")