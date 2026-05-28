from flask import Flask, render_template, request, redirect, url_for

import pickle
import numpy as np
import sqlite3
import json


# =========================================
# CREATE FLASK APP
# =========================================

app = Flask(__name__)


# =========================================
# CREATE DATABASE
# =========================================

conn = sqlite3.connect(

    'history.db',

    check_same_thread=False

)

cursor = conn.cursor()

cursor.execute(

    '''

    CREATE TABLE IF NOT EXISTS predictions (

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        contract_type TEXT,

        credit_amount REAL,

        age REAL,

        result TEXT,

        risk_score REAL,

        occupation_type TEXT

    )

    '''

)

conn.commit()


# =========================================
# LOAD TRAINED MODEL
# =========================================

model = pickle.load(

    open(

        'credit_fraud_model.pkl',

        'rb'

    )

)


# =========================================
# LOGIN PAGE
# =========================================

@app.route('/')

def login():

    return render_template(

        'login.html'

    )


# =========================================
# LOGIN VALIDATION
# =========================================

@app.route('/login', methods=['POST'])

def validate_login():

    username = request.form['username']

    password = request.form['password']


    if username == "admin" and password == "1234":

        return redirect(

            url_for('home')

        )

    else:

        return render_template(

            'login.html',

            error="Invalid Username or Password"

        )


# =========================================
# HOME PAGE
# =========================================

@app.route('/home')

def home():

    return render_template(

        'index.html'

    )


# =========================================
# PREDICTION ROUTE
# =========================================

@app.route('/predict', methods=['POST'])

def predict():

    # =========================================
    # GET VALUES FROM HTML FORM
    # =========================================

    contract_type = request.form['contract_type']

    age = float(

        request.form['age']

    )

    credit = float(

        request.form['credit']

    )

    income = float(

        request.form['income']

    )

    income_type = request.form['income_type']

    education = request.form['education']

    occupation_type = request.form['occupation_type']

    days_employed = float(

        request.form['days_employed']

    )


    # =========================================
    # VALIDATION
    # =========================================

    if age < 18 or age > 100:

        return render_template(

            'index.html',

            prediction_text="Invalid age entered"

        )


    # =========================================
    # MANUAL ENCODING
    # =========================================

    # CONTRACT TYPE

    if contract_type.lower() == "cash loans":

        contract_type = 0

    else:

        contract_type = 1


    # INCOME TYPE

    if income_type.lower() == "working":

        income_type = 0

    else:

        income_type = 1


    # EDUCATION TYPE

    if education.lower() == "higher education":

        education = 0

    else:

        education = 1


    # OCCUPATION TYPE

    if occupation_type.lower() in [

        'managers',
        'it staff',
        'accountants'

    ]:

        occupation_type = 0

    else:

        occupation_type = 1


    # =========================================
    # CREATE INPUT ARRAY
    # =========================================

    data = np.array([[

        contract_type,
        age,
        credit,
        income,
        income_type,
        education,
        occupation_type,
        days_employed

    ]])


    # =========================================
    # MODEL PREDICTION
    # =========================================

    prediction = model.predict(data)


    # =========================================
    # PROBABILITY SCORE
    # =========================================

    if hasattr(model, 'predict_proba'):

        probabilities = model.predict_proba(data)

        risk_score = round(

            probabilities[0][1] * 100,

            2

        )

    else:

        risk_score = 100 if prediction[0] == 1 else 0


    # =========================================
    # LOGICAL RISK ENGINE
    # =========================================

    risk_flags = 0


    # AGE CHECK

    if age > 75:

        risk_flags += 1


    # LOW INCOME

    if income < 20000:

        risk_flags += 1


    # CREDIT VS INCOME

    if credit > (income * 5):

        risk_flags += 1


    # EMPLOYMENT STABILITY

    if days_employed < 365:

        risk_flags += 1


    # =========================================
    # FINAL RESULT
    # =========================================

    if risk_flags >= 2:

        result = "⚠ High Risk Customer"

        risk_score = max(

            risk_score,

            85

        )

    else:

        if prediction[0] == 1:

            result = "⚠ High Risk Customer"

        else:

            result = "Low Risk Customer"


    # =========================================
    # SAVE TO DATABASE
    # =========================================

    conn = sqlite3.connect(

        'history.db'

    )

    cursor = conn.cursor()

    cursor.execute(

        """

        INSERT INTO predictions (

            contract_type,

            credit_amount,

            age,

            result,

            risk_score,

            occupation_type

        )

        VALUES (?, ?, ?, ?, ?, ?)

        """,

        (

            str(contract_type),

            float(credit),

            float(age),

            result,

            float(risk_score),

            str(request.form['occupation_type'])

        )

    )

    conn.commit()

    conn.close()


    # =========================================
    # RETURN RESULT
    # =========================================

    return render_template(

        'index.html',

        prediction_text=result,

        risk_score=risk_score

    )


# =========================================
# HISTORY PAGE
# =========================================

@app.route('/history')

def history():

    conn = sqlite3.connect(

        'history.db'

    )

    cursor = conn.cursor()

    cursor.execute(

        "SELECT * FROM predictions"

    )

    rows = cursor.fetchall()

    conn.close()

    return render_template(

        'history.html',

        predictions=rows

    )


# =========================================
# ANALYSIS PAGE
# =========================================

@app.route('/analysis')

def analysis():

    conn = sqlite3.connect(

        'history.db'

    )

    cursor = conn.cursor()


    # HIGH RISK VS LOW RISK

    cursor.execute(

        "SELECT result, COUNT(*) FROM predictions GROUP BY result"

    )

    results = cursor.fetchall()


    high_risk_count = 0

    low_risk_count = 0


    for res in results:

        if 'High Risk' in res[0]:

            high_risk_count += res[1]

        else:

            low_risk_count += res[1]


    # CREDIT AMOUNTS

    cursor.execute(

        "SELECT credit_amount FROM predictions"

    )

    credit_amounts = [

        row[0]

        for row in cursor.fetchall()

    ]

    conn.close()


    # LOAD MODEL METRICS

    try:

        with open(

            'model_metrics.json',

            'r'

        ) as f:

            metrics = json.load(f)

    except FileNotFoundError:

        metrics = None


    return render_template(

        'analysis.html',

        high_risk=high_risk_count,

        low_risk=low_risk_count,

        credit_amounts=credit_amounts,

        metrics=metrics

    )


# =========================================
# RUN FLASK APP
# =========================================

if __name__ == "__main__":

    app.run(

        debug=True

    )