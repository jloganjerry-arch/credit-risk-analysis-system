from flask import Flask, render_template, request, redirect, url_for, jsonify
import pickle
import numpy as np
import pandas as pd
import sqlite3
import json
import shap
import warnings
import matplotlib
import os
from dotenv import load_dotenv
import groq
from groq import Groq

matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()
# Note: google-genai client automatically picks up GEMINI_API_KEY from environment

# =========================================
# CREATE FLASK APP
# =========================================
app = Flask(__name__)

# =========================================
# CREATE DATABASE
# =========================================
conn = sqlite3.connect('history.db', check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        contract_type TEXT,
        credit_amount REAL,
        age REAL,
        result TEXT,
        risk_score REAL,
        occupation_type TEXT,
        risk_reason TEXT
    )
''')
# Alter table to add risk_reason if it doesn't exist from older versions
try:
    cursor.execute('ALTER TABLE predictions ADD COLUMN risk_reason TEXT')
except:
    pass
conn.commit()

# =========================================
# LOAD TRAINED MODEL & PROCESSORS
# =========================================
try:
    model = pickle.load(open('credit_fraud_model.pkl', 'rb'))
    label_encoders = pickle.load(open('label_encoders.pkl', 'rb'))
    imputer = pickle.load(open('imputer.pkl', 'rb'))
    explainer = pickle.load(open('shap_explainer.pkl', 'rb'))
except Exception as e:
    print(f"Warning: Model files missing or corrupt. Please run train_model.py first. Error: {e}")
    model, label_encoders, imputer, explainer = None, None, None, None

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

# =========================================
# AI ASSISTANT CHATBOT PAGE
# =========================================
@app.route('/chat')
def chat():
    return render_template('chat.html')

# =========================================
# API: AI CHAT
# =========================================
@app.route('/api/chat', methods=['POST'])
def api_chat():
    data = request.get_json()
    user_message = data.get('message', '')
    
    system_prompt = """You are a specialized Credit Risk Analysis Assistant.
You MUST ONLY answer questions related to:
- Credit Risk Analysis
- Risk Scores
- Prediction Results
- SHAP Explanations
- Model Features
- Credit Amount, Income, Occupation, Education, Employment Analysis
- Project Functionality

If a user asks an unrelated question (e.g. "Who won the IPL?", "Write a poem", "What is the capital of France?", etc.), you MUST decline by responding exactly with:
"I am a specialized Credit Risk Analysis Assistant and can only answer questions related to transaction risk assessment and prediction results."

Maintain a professional, banking-style tone at all times.
"""
    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        )
        return jsonify({"reply": response.choices[0].message.content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================================
# API: ANALYZE WITH AI
# =========================================
@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    data = request.get_json()
    
    prompt = f"""You are an expert AI Credit Risk Analyst. Analyze the following prediction data and generate a structured risk assessment report.

Prediction Data:
- Risk Category: {data.get('risk_category')}
- Risk Score: {data.get('risk_score')}%
- Contract Type: {data.get('contract_type')}
- Credit Amount: {data.get('credit')}
- Total Income: {data.get('income')}
- Age Profile: {data.get('age')} years
- Employment History: {data.get('days_employed')} days
- Occupation: {data.get('occupation')}
- Education: {data.get('education')}
- SHAP/Rule Insights: {data.get('risk_reason')}

Generate a structured output exactly containing these sections (use Markdown):
### AI Risk Assessment
[Provide a clear summary sentence like "This applicant was classified as <Category> because..."]

### Key Risk Factors
[Bullet points of the main factors driving this score]

### SHAP Insights
[Explain how the AI weighted the specific features (SHAP values)]

### Potential Concerns
[Any red flags or specific warnings]

### Recommendation
[Actionable recommendation for the loan officer (e.g., approve, manual verification, reject)]

Maintain a highly professional, banking-style tone.
"""
    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are an expert AI Credit Risk Analyst."},
                {"role": "user", "content": prompt}
            ]
        )
        return jsonify({"analysis": response.choices[0].message.content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================================
# LOGIN PAGE
# =========================================
@app.route('/')
def login():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def validate_login():
    username = request.form['username']
    password = request.form['password']
    if username == "admin" and password == "1234":
        return redirect(url_for('home'))
    else:
        return render_template('login.html', error="Invalid Username or Password")

# =========================================
# HOME PAGE
# =========================================
@app.route('/home')
def home():
    return render_template('index.html')

# =========================================
# PREDICTION ROUTE
# =========================================
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', prediction_text="Model not trained yet. Please run training script.")

    # Get raw inputs
    raw_data = {
        'NAME_CONTRACT_TYPE': request.form['contract_type'],
        'DAYS_BIRTH': float(request.form['age']) * 365,  # Convert back to raw DAYS for imputer match
        'AMT_CREDIT': float(request.form['credit']),
        'AMT_INCOME_TOTAL': float(request.form['income']),
        'NAME_INCOME_TYPE': request.form['income_type'],
        'NAME_EDUCATION_TYPE': request.form['education'],
        'OCCUPATION_TYPE': request.form['occupation_type'],
        'DAYS_EMPLOYED': float(request.form['days_employed'])
    }
    
    age = float(request.form['age'])
    credit = raw_data['AMT_CREDIT']
    income = raw_data['AMT_INCOME_TOTAL']
    days_employed = raw_data['DAYS_EMPLOYED']

    # Validation
    if age < 18 or age > 100:
        return render_template('index.html', prediction_text="Invalid age entered")

    # Dynamic Preprocessing matching training script
    input_df = pd.DataFrame([raw_data])
    
    # Impute
    input_df_imputed = pd.DataFrame(imputer.transform(input_df), columns=features)
    
    # Feature Engineering
    input_df_imputed['DAYS_BIRTH'] = abs(input_df_imputed['DAYS_BIRTH'].astype(float)) / 365
    input_df_imputed['DAYS_EMPLOYED'] = abs(input_df_imputed['DAYS_EMPLOYED'].astype(float))
    
    # Encode categorical
    for col in ['NAME_CONTRACT_TYPE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE']:
        val = str(input_df_imputed[col].iloc[0])
        # Handle unseen labels gracefully by assigning to a known class (or 0)
        if val in label_encoders[col].classes_:
            input_df_imputed[col] = label_encoders[col].transform([val])[0]
        else:
            input_df_imputed[col] = 0
            
    # Convert all to float
    for col in input_df_imputed.columns:
        input_df_imputed[col] = input_df_imputed[col].astype(float)

    # ML Model Prediction
    prediction = model.predict(input_df_imputed)
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(input_df_imputed)
        risk_score = round(probabilities[0][1] * 100, 2)
    else:
        risk_score = 100.0 if prediction[0] == 1 else 0.0
        
    is_high_risk = (prediction[0] == 1)

    # =========================================
    # RULE-BASED VALIDATION OVERRIDES
    # =========================================
    rule_reason = ""
    if age < 25 and income < 30000 and credit > 250000:
        is_high_risk = True
        risk_score = max(risk_score, 98.0)
        rule_reason = "Flagged due to young age, low income, and excessively high requested credit."
    elif str(raw_data['NAME_INCOME_TYPE']).lower() == 'student' and credit > 100000:
        is_high_risk = True
        risk_score = max(risk_score, 99.0)
        rule_reason = "Flagged due to unusually high loan request for a student."
    elif days_employed < 180 and credit > 200000:
        is_high_risk = True
        risk_score = max(risk_score, 95.0)
        rule_reason = "Flagged due to very low employment history combined with high credit requirement."
    elif income > 0 and (credit / income) > 10:
        is_high_risk = True
        risk_score = max(risk_score, 96.0)
        rule_reason = "Flagged due to dangerously high credit-to-income ratio (>10x)."

    # =========================================
    # RISK CATEGORY MAPPING (3-Tier)
    # =========================================
    if risk_score >= 70:
        risk_category = "High Risk"
    elif risk_score >= 40:
        risk_category = "Medium Risk"
    else:
        risk_category = "Low Risk"

    # Rule-based overrides override category
    if rule_reason:
        risk_category = "High Risk"

    # =========================================
    # SHAP HUMAN-READABLE EXPLANATION
    # =========================================
    shap_vals = explainer.shap_values(input_df_imputed)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    
    base_value = explainer.expected_value
    if isinstance(base_value, list):
        base_value = base_value[1]
    elif isinstance(base_value, np.ndarray) and len(base_value) > 1:
        base_value = base_value[1]
        
    explanation = shap.Explanation(
        values=shap_vals[0],
        base_values=base_value,
        data=input_df_imputed.iloc[0].values,
        feature_names=features
    )
    
    # Generate and save Waterfall Plot
    plt.figure(figsize=(7, 4))
    shap.plots.waterfall(explanation, show=False)
    plt.tight_layout()
    plt.savefig('static/local_shap.png', bbox_inches='tight', dpi=150)
    plt.close()

    # Sort features by absolute impact for textual reason
    impacts = shap_vals[0]
    feature_impacts = [(features[i], impacts[i]) for i in range(len(features))]
    feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
    
    top_pushing_risk = [f[0] for f in feature_impacts if f[1] > 0][:3]
    top_lowering_risk = [f[0] for f in feature_impacts if f[1] < 0][:3]
    
    # Map raw features to human names and explanations
    feature_name_map = {
        'NAME_CONTRACT_TYPE': 'Contract type',
        'DAYS_BIRTH': 'Age profile',
        'AMT_CREDIT': 'Credit amount',
        'AMT_INCOME_TOTAL': 'Total income',
        'NAME_INCOME_TYPE': 'Income source',
        'NAME_EDUCATION_TYPE': 'Education level',
        'OCCUPATION_TYPE': 'Occupation',
        'DAYS_EMPLOYED': 'Employment history'
    }
    
    risk_reason_html = "<strong>💡 AI Explanation</strong><br><br>"
    
    if rule_reason:
        risk_reason_html += f"This application was classified as HIGH RISK due to a critical rule violation:<br><br><ul><li>{rule_reason}</li></ul>"
        risk_reason_html += "<br><strong>Recommendation:</strong><br>Immediate rejection or manual verification required."
    else:
        if risk_category == "High Risk":
            risk_reason_html += "This application was classified as HIGH RISK because:<br><br><ul>"
            for f in top_pushing_risk:
                risk_reason_html += f"<li>{feature_name_map.get(f, f)} contributes negatively to repayment confidence.</li>"
            risk_reason_html += "</ul><br><strong>Recommendation:</strong><br>Manual verification is advised before approval."
            
        elif risk_category == "Medium Risk":
            risk_reason_html += "This application was classified as MEDIUM RISK. Key factors include:<br><br><ul>"
            if top_pushing_risk:
                risk_reason_html += f"<li>{feature_name_map.get(top_pushing_risk[0], top_pushing_risk[0])} requires attention.</li>"
            if top_lowering_risk:
                risk_reason_html += f"<li>{feature_name_map.get(top_lowering_risk[0], top_lowering_risk[0])} provides some stability.</li>"
            risk_reason_html += "</ul><br><strong>Recommendation:</strong><br>Proceed with caution and standard verification."
            
        else:
            risk_reason_html += "This application was classified as LOW RISK because:<br><br><ul>"
            for f in top_lowering_risk:
                risk_reason_html += f"<li>{feature_name_map.get(f, f)} strongly supports repayment capability.</li>"
            risk_reason_html += "</ul><br><strong>Recommendation:</strong><br>Standard processing approved."

    # =========================================
    # SAVE TO DATABASE
    # =========================================
    conn = sqlite3.connect('history.db')
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (contract_type, credit_amount, age, result, risk_score, occupation_type, risk_reason)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (str(raw_data['NAME_CONTRACT_TYPE']), float(credit), float(age), risk_category, float(risk_score), str(raw_data['OCCUPATION_TYPE']), risk_reason_html))
    conn.commit()
    conn.close()

    return render_template('index.html', prediction_text=risk_category, risk_score=risk_score, risk_reason=risk_reason_html, show_results=True)

# =========================================
# HISTORY PAGE
# =========================================
@app.route('/history')
def history():
    conn = sqlite3.connect('history.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return render_template('history.html', predictions=rows)

# =========================================
# ANALYSIS PAGE
# =========================================
@app.route('/analysis')
def analysis():
    conn = sqlite3.connect('history.db')
    cursor = conn.cursor()
    cursor.execute("SELECT result, COUNT(*) FROM predictions GROUP BY result")
    results = cursor.fetchall()
    high_risk_count = 0
    low_risk_count = 0
    for res in results:
        if 'High Risk' in res[0]:
            high_risk_count += res[1]
        else:
            low_risk_count += res[1]

    cursor.execute("SELECT credit_amount FROM predictions")
    credit_amounts = [row[0] for row in cursor.fetchall()]
    conn.close()

    try:
        with open('model_metrics.json', 'r') as f:
            metrics = json.load(f)
    except FileNotFoundError:
        metrics = None

    return render_template('analysis.html', high_risk=high_risk_count, low_risk=low_risk_count, credit_amounts=credit_amounts, metrics=metrics)

# =========================================
# RUN FLASK APP
# =========================================
if __name__ == "__main__":
    app.run(debug=True)