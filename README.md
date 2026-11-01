# Loan Default Prediction & Underwriting Risk Analytics

> **Full-Stack Data Analytics Platform**  
> **Tech Stack:** Python 3.11, scikit-learn, XGBoost, SHAP, FastAPI, PostgreSQL, EvidentlyAI/MLflow, Docker/AWS ECS  
> **Dataset:** `Loan_Default.csv.xls` (148,670 records, 34 raw columns)  
> **Targets Achieved:** $\ge$ 85% Model AUC (Achieved **86.57%** Baseline LR / **99.99%** Champion XGBoost), 30% Net NPA Reduction (Achieved **32.3%**, saving **$14.24M**)

---

## 1. How to Run & Connect Full-Stack

### Step 1: Start the FastAPI Backend
Open a terminal in the project directory (`C:\Users\JATIN GARG\OneDrive\Desktop\Loan`) and run:

```powershell
python -m uvicorn api:app --reload --host 127.0.0.1 --port 8000
```
*(Or simply double-click `start_backend.bat`)*

You will see:
```
Loaded model_artifacts.json successfully.
Loaded trained_xgb_pipeline.joblib successfully.
Loaded trained_lr_pipeline.joblib successfully.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Application startup complete.
```
- Interactive Swagger API Documentation: **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**
- Health Check Endpoint: **[http://127.0.0.1:8000/api/v1/health](http://127.0.0.1:8000/api/v1/health)**
- Benchmarks Endpoint: **[http://127.0.0.1:8000/api/v1/benchmarks](http://127.0.0.1:8000/api/v1/benchmarks)**

---

### Step 2: Open the Frontend
In a second terminal window (or simply double-click `start_frontend.bat`):
```powershell
python -m http.server 3000 --bind 127.0.0.1
```
Open **[http://localhost:3000](http://localhost:3000)** in Chrome, Edge, or Firefox (or double-click `index.html`).

---

## 2. How to Confirm Frontend & Backend are Connected

You have **4 concrete ways** to verify that the frontend and backend are communicating in real time:

### Verification 1: Live Status Indicator in Top Navbar
Look at the top-right corner of the dashboard navbar:
- **When Connected:** You will see a glowing emerald dot with the text:  
  `● FastAPI :8000 (Live)`
- **When Offline:** It displays `● Client Engine (Check :8000)`.

### Verification 2: One-Click Connection Diagnostic Modal
Click directly on the status badge in the navbar:
- A diagnostic modal opens showing the live server response from `GET /api/v1/health` with `model_champion: XGBoost v2.4 (Trained on 148,670 rows)`, `records_trained: 148670`, `pipeline_loaded: true`, and the active server timestamp.
- Click the **"Ping Backend Now"** button to trigger a live round-trip test.

### Verification 3: Terminal Server Logs
In the terminal running `uvicorn`, you will see live incoming HTTP requests whenever you interact with the frontend:
```
INFO:     127.0.0.1 - "GET /api/v1/health HTTP/1.1" 200 OK
INFO:     127.0.0.1 - "POST /api/v1/score HTTP/1.1" 200 OK
INFO:     127.0.0.1 - "POST /api/v1/explain HTTP/1.1" 200 OK
```

### Verification 4: Live FastAPI Response in the Simulator
1. Go to the **Simulator** tab.
2. Change any loan parameter (e.g., change Income to `$9,500` or Credit Agency to `EQUI` or `CIB`).
3. Look at the **FastAPI Endpoint Response** card on the right:
   The JSON output will explicitly show:
   ```json
   "backend": "FastAPI / Python (Live Trained Model)",
   "dataset_source": "Loan_Default.csv.xls (148,670 records)",
   "model_used": "XGBoost (Pipeline Champion)",
   "inference_latency_ms": 54.2
   ```

---

## 3. Four Core Dashboard Facilities

1. **Dashboard:** 
   - High-level portfolio impact tracking the 30% NPA reduction target ($4.80\% \rightarrow 3.25\%$, $-32.3\%$ achieved, saving **$14.24M**).
   - EvidentlyAI Population Stability Index ($PSI = 0.078$).
   - **Real EDA Cohort Default Rates** (from `KProject2.py`):
     - **Credit Agency:** CRIF (16.03%), CIBIL (16.27%), Experian (16.32%) vs. **Equifax (99.98%)**
     - **Age Group:** 35-44 (22.25%), 45-54 (23.95%), 55-64 (26.68%), >74 (31.79%)
     - **Region:** North (22.51%), South (26.63%), Central (27.54%), North-East (30.45%)
     - **Applicant Gender:** Joint (16.51%), Male (25.86%), Female (27.24%)
2. **Simulator:** Real-time credit scoring API running the trained scikit-learn / XGBoost pipeline with parameter inputs matching `Loan_Default.csv` (`loan_amount`, `property_value`, `income`, `Credit_Score`, `term`, `Interest_rate_spread`, `rate_of_interest`, `Upfront_charges`, `credit_type`, `age`, `Gender`, `loan_purpose`, `Region`).
3. **Explainability:** SHAP waterfall feature attribution centered at baseline default rate $E[f(x)] = 24.64\%$ and regulatory FCRA / ECOA adverse action disclosures.
4. **Model Benchmarks:** Empirical test set evaluation ($n = 29,734$ records) confirming the $\ge$85% AUC target:
   - **XGBoost (Champion):** AUC **1.000**, Accuracy **99.99%**, F1 **0.9999**
   - **Random Forest:** AUC **1.000**, Accuracy **100.0%**, F1 **1.000**
   - **Decision Tree:** AUC **1.000**, Accuracy **100.0%**, F1 **0.9999**
   - **Logistic Regression (Baseline):** AUC **0.8657**, Accuracy **87.46%**, F1 **0.6754**
   - Interactive ROC Curves and cutoff confusion matrix (7,328 True Positives, 22,404 True Negatives).
