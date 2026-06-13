import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import shap
import warnings
import os
import time

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings('ignore')

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
print(f"Class imbalance (Neg/Pos): {imbalance_ratio:.2f}")

print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Use a 20% subset of training data for hyperparameter tuning to save time
print("Creating tuning subset...")
X_tune, _, y_tune, _ = train_test_split(X_train, y_train, train_size=0.2, random_state=42, stratify=y_train)
print(f"Tuning set size: {len(X_tune)} rows. Full train set size: {len(X_train)} rows.")

# Model definitions and parameter grids
# 1. CatBoost
cat_model = CatBoostClassifier(random_state=42, verbose=0, scale_pos_weight=imbalance_ratio)
cat_params = {
    'iterations': [100, 200],
    'depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1]
}

# 2. LightGBM
lgbm_model = LGBMClassifier(random_state=42, scale_pos_weight=imbalance_ratio, n_jobs=-1, verbose=-1)
lgbm_params = {
    'n_estimators': [100, 200],
    'max_depth': [4, 6, -1],
    'learning_rate': [0.05, 0.1],
    'num_leaves': [31, 63]
}

# 3. MLPClassifier (Neural Network)
# MLP requires scaling, so we put it in a Pipeline
mlp_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(random_state=42, max_iter=100, early_stopping=True))
])
mlp_params = {
    'mlp__hidden_layer_sizes': [(50,), (100,)],
    'mlp__activation': ['relu', 'tanh'],
    'mlp__alpha': [0.0001, 0.001],
    'mlp__learning_rate_init': [0.001, 0.01]
}

models_to_compare = {
    "CatBoost": (cat_model, cat_params),
    "LightGBM": (lgbm_model, lgbm_params),
    "MLPClassifier": (mlp_pipeline, mlp_params)
}

best_models = {}
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

print("\n--- Starting Hyperparameter Tuning ---")
for name, (model, params) in models_to_compare.items():
    print(f"\nTuning {name}...")
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=params,
        n_iter=5, # Keep iterations low for speed
        scoring='roc_auc',
        cv=cv,
        random_state=42,
        n_jobs=-1 if name != "CatBoost" else 1 # CatBoost handles threads internally
    )
    search.fit(X_tune, y_tune)
    best_models[name] = search.best_estimator_
    print(f"Best {name} AUC on tune set: {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")

print("\n--- Training Best Models on Full Dataset ---")
model_results = []
trained_models = {}

for name, model in best_models.items():
    print(f"Training {name} on full {len(X_train)} rows...")
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    
    t1 = time.time()
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    pred_time = time.time() - t1
    
    # Train AUC to check overfitting
    y_train_prob = model.predict_proba(X_train)[:, 1]
    train_auc = roc_auc_score(y_train, y_train_prob)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    model_results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": roc_auc,
        "Train ROC-AUC": train_auc,
        "Overfit Delta": train_auc - roc_auc,
        "Train Time (s)": train_time,
        "Predict Time (s)": pred_time
    })
    trained_models[name] = model

results_df = pd.DataFrame(model_results)
print("\n--- MODEL COMPARISON TABLE ---")
print(results_df.to_string(index=False))
results_df.to_csv("comparison_report.csv", index=False)

# Select best model based on Test ROC-AUC
best_row = results_df.sort_values(by="ROC-AUC", ascending=False).iloc[0]
best_name = best_row["Model"]
final_best_model = trained_models[best_name]

print(f"\n===================================")
print(f"BEST MODEL SELECTED: {best_name} (Test ROC-AUC: {best_row['ROC-AUC']:.4f})")
print(f"===================================")

y_pred_best = final_best_model.predict(X_test)
y_prob_best = final_best_model.predict_proba(X_test)[:, 1]

metrics = {
    "best_model": best_name,
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
    pickle.dump(final_best_model, f)
with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)
with open("imputer.pkl", "wb") as f:
    pickle.dump(imputer, f)

# Determine if we need to save the StandardScaler for MLP
if best_name == "MLPClassifier":
    scaler = final_best_model.named_steps['scaler']
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    metrics["requires_scaling"] = True
else:
    metrics["requires_scaling"] = False

# SHAP EXPLAINABILITY
print("\nGenerating SHAP explanations for Best Model...")
shap_sample = shap.sample(X_train, 1000)

if best_name in ["CatBoost", "LightGBM"]:
    explainer = shap.TreeExplainer(final_best_model)
    shap_values = explainer.shap_values(shap_sample)
    if isinstance(shap_values, list) and len(shap_values) > 1:
        shap_values = shap_values[1]
else:
    mlp = final_best_model.named_steps['mlp']
    scaler = final_best_model.named_steps['scaler']
    scaled_sample = scaler.transform(shap_sample)
    
    def predict_fn(X_sub):
        return mlp.predict_proba(X_sub)[:, 1]
    
    background = shap.kmeans(scaler.transform(X_train), 50)
    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(scaled_sample, nsamples=100)

with open("shap_explainer.pkl", "wb") as f:
    pickle.dump(explainer, f)

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
if best_name == "MLPClassifier":
    shap.summary_plot(shap_values, scaled_sample, feature_names=features, show=False)
else:
    shap.summary_plot(shap_values, shap_sample, feature_names=features, show=False)
plt.savefig('static/shap_summary.png', bbox_inches='tight', dpi=300)
plt.close()

# Generate ROC Curve
print("Saving ROC Curve...")
fpr, tpr, _ = roc_curve(y_test, y_prob_best)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {best_row["ROC-AUC"]:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'Receiver Operating Characteristic - {best_name}')
plt.legend(loc="lower right")
plt.savefig('static/roc_curve.png', bbox_inches='tight', dpi=300)
plt.close()

print("\n===================================")
print("MODEL TRAINING & EXPLAINABILITY COMPLETED")
print("===================================")