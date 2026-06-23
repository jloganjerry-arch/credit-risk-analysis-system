# Credit Risk Analysis & Transaction Analyzer

[![Deploy to Render](https://img.shields.io/badge/Deploy%20to-Render-4353ff?style=for-the-badge&logo=render&logoColor=white)](https://transaction-risk-analysis.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-1.0+-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)

An enterprise-grade, ML-powered **Credit Risk Analysis & Intelligence System**. The application uses an ensemble of CatBoost, LightGBM, and XGBoost models to calculate credit default probability, combined with **Explainable AI (SHAP)** narratives, a **What-If Scenario Simulator**, a **Compliance Audit Log**, and **Model Drift Monitoring**.

### 🔗 Live Production Demo: [https://transaction-risk-analysis.onrender.com](https://transaction-risk-analysis.onrender.com)

---

## 🌟 Key Features

### 1. 🔍 Transaction Analyzer Dashboard
- **Inputs Panel:** Multi-factor applicant screening (Contract type, total income, credit request, occupation, age, employment length).
- **SVG Radial Gauge:** Glow-based dynamic circular meter visualizing applicant risk probability (Low, Medium, High Risk).
- **Staggered Animations:** Seamless fade-and-slide layouts when predictions render.

### 2. 🔬 Interactive What-If Simulator
- Divided into a two-column slider grid for parameter adjustments.
- Instantly recalculates predicted risk probability via the `/api/simulate` endpoint when modifying key variables (Income, Credit, Age, Days Employed) without overwriting the officially saved transaction.

### 3. 📊 Explainable AI (SHAP)
- **Local Interpretability:** View-zoomable local SHAP waterfall charts showing how much each input pushed the risk score up or down.
- **Global Dashboard:** Dedicated page displaying average SHAP feature importance bars, beeswarm distribution plots, and core model evaluation metrics (Accuracy, precision, recall, F1, AUC).

### 4. 📋 Compliance Audit Log (Admin Only)
- Track critical operations (logins, logouts, risk predictions, PDF downloads, and email logs) with timestamps, usernames, details, and client IP addresses.
- Full text search, action filter badges, and direct **Export CSV** download utility.

### 5. 📈 Model Drift Monitoring (Admin & Manager)
- Measures statistical divergence between current runtime transactions and the training baseline.
- Tracks feature-level **PSI (Population Stability Index)** to flag warning/drift states (PSI > 0.2) before models become stale.

### 6. 🗂️ Batch Processing & Reporting
- Batch prediction via CSV file uploads.
- PDF Report generation powered by **ReportLab** containing credit summary tables and AI risk narratives.
- SMTP-based email distribution of reports.

---

## 🛠️ Technology Stack

* **Backend:** Python, Flask, SQLite (Development) / PostgreSQL (Production)
* **Machine Learning:** CatBoost, LightGBM, XGBoost, Scikit-Learn, SHAP
* **Frontend:** Vanilla HTML5, CSS3 (Modern Dark Theme, CSS Variables, Lucide SVG Icons), JavaScript (ES6, SVG Ring Gauges)
* **Reporting:** ReportLab (PDF), openpyxl (Excel)
* **Deployment:** Gunicorn, Render (`Procfile`)

---

## 🚀 Local Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/jloganjerry-arch/credit-risk-analysis-system.git
cd credit-risk-analysis-system
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
SECRET_KEY=your_flask_secret_key_here
# DATABASE_URL=optional_postgres_url
```

### 4. Run the Application
```bash
python app.py
```
Open `http://127.0.0.1:5000` in your web browser.

---

## ☁️ Deployment

This repository is ready for immediate deployment on **Render** or **Heroku**:
- **Procfile:** Configured to run `web: gunicorn app:app`.
- **requirements.txt:** Lists all Python dependencies (including `openpyxl` and `psycopg2-binary` for database integration).

When deploying to Render:
1. Connect this repository to your Render Web Service.
2. Link the environment variable `GROQ_API_KEY` in the Render configuration settings.
3. Deploy!
