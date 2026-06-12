from flask import Flask, render_template, request

import pickle

import numpy as np

app = Flask(__name__)

# LOAD MODEL

model = pickle.load(

    open(

        'credit_fraud_model.pkl',

        'rb'
    )
)

# HOME PAGE

@app.route('/')

def home():

    return render_template(

        'index.html'
    )

# PREDICTION

@app.route(

    '/predict',

    methods=['POST']
)

def predict():

    values = [

        float(x)

        for x in request.form.values()
    ]

    data = np.array([values])

    prediction = model.predict(data)

    if prediction[0] == 1:

        result = "⚠ Fraud Detected"

    else:

        result = "✅ Non-Fraud Transaction"

    return render_template(

        'index.html',

        prediction_text=result
    )

if __name__ == "__main__":

    app.run(debug=True)