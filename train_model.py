import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import shap
import warnings
import os

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

print("Loading dataset...")
df = pd.read_csv("application_data.csv", usecols=[
    'TARGET', 'NAME_CONTRACT_TYPE', 'DAYS_BIRTH', 'AMT_CREDIT', 
    'AMT_INCOME_TOTAL', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 
    'OCCUPATION_TYPE', 'DAYS_EMPLOYED'
])

print("Preprocessing...")
# Handle specific anomalies
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

# Convert all columns to float explicitly to avoid XGBoost/CatBoost errors
for col in X.columns:
    X[col] = X[col].astype(float)

print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Defining models...")
models = {
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
    "CatBoost": CatBoostClassifier(iterations=200, depth=6, learning_rate=0.1, random_state=42, verbose=0),
    "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='logloss')
}

best_model_name = ""
best_model = None
best_auc = -1
model_results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    model_results[name] = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": roc_auc
    }
    
    print(f"{name} Results: Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}, AUC: {roc_auc:.4f}")
    
    # In fraud detection, ROC-AUC is typically best for generalization
    if roc_auc > best_auc:
        best_auc = roc_auc
        best_model_name = name
        best_model = model

print(f"\n===================================")
print(f"BEST MODEL SELECTED: {best_model_name} (ROC-AUC: {best_auc:.4f})")
print(f"===================================")

# Evaluate Best Model fully
y_pred_best = best_model.predict(X_test)
y_prob_best = best_model.predict_proba(X_test)[:, 1]

# Save Metrics
metrics = {
    "best_model": best_model_name,
    "accuracy": accuracy_score(y_test, y_pred_best),
    "precision": precision_score(y_test, y_pred_best, zero_division=0),
    "recall": recall_score(y_test, y_pred_best, zero_division=0),
    "f1_score": f1_score(y_test, y_pred_best, zero_division=0),
    "roc_auc": roc_auc_score(y_test, y_prob_best),
    "confusion_matrix": confusion_matrix(y_test, y_pred_best).tolist(),
    "features": features
}

print("\nSaving Models and Processors...")
with open("credit_fraud_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)
with open("imputer.pkl", "wb") as f:
    pickle.dump(imputer, f)

# SHAP EXPLAINABILITY
print("\nGenerating SHAP explanations for Best Model...")

# Sample data for background to speed up SHAP
shap_sample = shap.sample(X_train, 1000)
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(shap_sample)

# If multi-class output, take the positive class
if isinstance(shap_values, list):
    shap_values = shap_values[1]

# Save the explainer itself so the Flask app can load it and run local explanations
with open("shap_explainer.pkl", "wb") as f:
    pickle.dump(explainer, f)

# Feature importance based on SHAP
shap_sum = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame([features, shap_sum.tolist()]).T
importance_df.columns = ['feature-name', 'shap-importance']
importance_df = importance_df.sort_values('shap-importance', ascending=False)
metrics["feature_importances"] = importance_df['shap-importance'].tolist()
metrics["features_sorted"] = importance_df['feature-name'].tolist()

with open("model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Saving SHAP Summary Plot...")
if not os.path.exists('static'):
    os.makedirs('static')
    
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, shap_sample, show=False)
plt.savefig('static/shap_summary.png', bbox_inches='tight', dpi=300)
plt.close()

# Generate ROC Curve
print("Saving ROC Curve...")
fpr, tpr, _ = roc_curve(y_test, y_prob_best)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {best_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'Receiver Operating Characteristic - {best_model_name}')
plt.legend(loc="lower right")
plt.savefig('static/roc_curve.png', bbox_inches='tight', dpi=300)
plt.close()

print("\n===================================")
print("MODEL TRAINING & EXPLAINABILITY COMPLETED")
print("===================================")