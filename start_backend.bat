@echo off
title Loan Default Prediction - FastAPI Backend Server
echo =====================================================================
echo  Starting FastAPI Backend Server...
echo  Technology: Python 3.11, scikit-learn, XGBoost, SHAP, FastAPI
echo  API URL:    http://127.0.0.1:8000
echo  Swagger UI: http://127.0.0.1:8000/docs
echo =====================================================================
python -m uvicorn api:app --reload --host 127.0.0.1 --port 8000
pause

