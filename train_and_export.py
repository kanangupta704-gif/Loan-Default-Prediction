"""
Train & Export Script based on KProject2.py
Processes Loan_Default.csv.xls, trains models, extracts real dataset analytics & exports to model_artifacts.json and trained_pipeline.joblib
"""

import pandas as pd
import numpy as np
import json
import joblib
import time
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix

csv_filename = "Loan_Default.csv" if os.path.exists("Loan_Default.csv") else "Loan_Default.csv.xls"
print(f"Reading dataset from {csv_filename}...")
df = pd.read_csv(csv_filename)
print(f"Dataset shape: {df.shape}")

# Preprocessing from notebook
df = df.drop_duplicates()
df['Gender'] = df['Gender'].replace('Sex Not Available', np.nan)
df = df.drop(columns=['ID', 'year'])

# Collect EDA statistics
total_records = len(df)
default_counts = df['Status'].value_counts().to_dict()
default_pct = (df['Status'].value_counts(normalize=True) * 100).to_dict()

# Sub-segment default rates from notebook
default_by_gender = (pd.crosstab(df['Gender'].fillna('Missing'), df['Status'], normalize='index')[1] * 100).to_dict()
age_default = (pd.crosstab(df['age'].fillna('Missing'), df['Status'], normalize='index')[1] * 100).to_dict()
credit_default = (pd.crosstab(df['credit_type'].fillna('Missing'), df['Status'], normalize='index')[1] * 100).to_dict()
purpose_default = (pd.crosstab(df['loan_purpose'].fillna('Missing'), df['Status'], normalize='index')[1] * 100).to_dict()
region_default = (pd.crosstab(df['Region'].fillna('Missing'), df['Status'], normalize='index')[1] * 100).to_dict()

# Medians for numerical features
num_medians = {
    col: float(df[col].median())
    for col in ['loan_amount','rate_of_interest','Interest_rate_spread','Upfront_charges','term','property_value','income','Credit_Score','LTV','dtir1']
    if col in df.columns
}

# Categorical modes
cat_modes = {
    col: str(df[col].mode()[0])
    for col in df.select_dtypes(include='object').columns
}

print(f"Status distribution: {default_pct}")

# Prepare features and target
X = df.drop('Status', axis=1)
y = df['Status']

# Train/Test Split (stratified)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

numerical_columns = list(X_train.select_dtypes(include=np.number).columns)
categorical_columns = list(X_train.select_dtypes(include='object').columns)

print(f"Numerical columns: {numerical_columns}")
print(f"Categorical columns: {categorical_columns}")

numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('numeric', numeric_pipeline, numerical_columns),
    ('categorical', categorical_pipeline, categorical_columns)
])

# For fast training on full 148k dataset, train LogisticRegression, DecisionTree, and XGBoost
print("Training Logistic Regression...")
lr_model = Pipeline([('preprocessor', preprocessor), ('model', LogisticRegression(max_iter=1000, random_state=42))])
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
y_prob_lr = lr_model.predict_proba(X_test)[:, 1]

print("Training Decision Tree...")
dt_model = Pipeline([('preprocessor', preprocessor), ('model', DecisionTreeClassifier(random_state=42, max_depth=10))])
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
y_prob_dt = dt_model.predict_proba(X_test)[:, 1]

print("Training XGBoost...")
xgb_model = Pipeline([('preprocessor', preprocessor), ('model', XGBClassifier(n_estimators=100, learning_rate=0.08, max_depth=6, random_state=42, eval_metric='logloss', n_jobs=-1))])
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

# Also Random Forest on sample or estimators=50 for speed
print("Training Random Forest...")
rf_model = Pipeline([('preprocessor', preprocessor), ('model', RandomForestClassifier(n_estimators=60, max_depth=12, random_state=42, n_jobs=-1))])
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

# Evaluate all
models_eval = [
    ("Logistic Regression", lr_model, y_pred_lr, y_prob_lr, "BASELINE"),
    ("Decision Tree", dt_model, y_pred_dt, y_prob_dt, "BENCHMARK"),
    ("Random Forest", rf_model, y_pred_rf, y_prob_rf, "CHALLENGER"),
    ("XGBoost", xgb_model, y_pred_xgb, y_prob_xgb, "CHAMPION")
]

results_list = []
roc_curves = {}

for name, model, pred, prob, role in models_eval:
    acc = float(accuracy_score(y_test, pred))
    prec = float(precision_score(y_test, pred, zero_division=0))
    rec = float(recall_score(y_test, pred, zero_division=0))
    f1 = float(f1_score(y_test, pred, zero_division=0))
    auc = float(roc_auc_score(y_test, prob))
    
    fpr, tpr, _ = roc_curve(y_test, prob)
    # Downsample ROC points for clean frontend rendering
    step = max(1, len(fpr) // 25)
    sampled_roc = [{"fpr": round(float(fpr[i]), 3), "tpr": round(float(tpr[i]), 3)} for i in range(0, len(fpr), step)]
    if sampled_roc[-1]["fpr"] != 1.0:
        sampled_roc.append({"fpr": 1.0, "tpr": 1.0})
    
    roc_curves[name.lower().replace(" ", "_")] = sampled_roc

    results_list.append({
        "name": name,
        "role": role,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "auc": round(auc, 4)
    })
    print(f"{name} -> AUC: {auc:.4f}, Acc: {acc:.4f}, F1: {f1:.4f}")

# Extract XGBoost Feature Importances
feat_names = xgb_model.named_steps['preprocessor'].get_feature_names_out()
feat_importances = xgb_model.named_steps['model'].feature_importances_
top_feat_df = pd.DataFrame({'feature': feat_names, 'importance': feat_importances}).sort_values('importance', ascending=False)
top_15_features = [
    {
        "feature": row['feature'].replace('numeric__', '').replace('categorical__', ''),
        "importance": round(float(row['importance']), 4)
    }
    for _, row in top_feat_df.head(15).iterrows()
]

# Calculate confusion matrix for XGBoost at default threshold 0.5 and 0.25
cm_xgb_50 = confusion_matrix(y_test, y_pred_xgb)
tp = int(cm_xgb_50[1][1])
fn = int(cm_xgb_50[1][0])
fp = int(cm_xgb_50[0][1])
tn = int(cm_xgb_50[0][0])

artifacts = {
    "dataset_summary": {
        "total_records": total_records,
        "default_count": int(default_counts.get(1, 0)),
        "non_default_count": int(default_counts.get(0, 0)),
        "default_rate_pct": round(float(default_pct.get(1, 0)), 2),
        "features_count": X.shape[1],
        "test_size": len(y_test)
    },
    "model_benchmarks": results_list,
    "top_features": top_15_features,
    "roc_curves": roc_curves,
    "eda_crosstabs": {
        "gender": {k: round(v, 2) for k, v in default_by_gender.items()},
        "age": {k: round(v, 2) for k, v in age_default.items()},
        "credit_type": {k: round(v, 2) for k, v in credit_default.items()},
        "loan_purpose": {k: round(v, 2) for k, v in purpose_default.items()},
        "region": {k: round(v, 2) for k, v in region_default.items()}
    },
    "feature_medians": num_medians,
    "feature_modes": cat_modes,
    "confusion_matrix_xgb": {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn
    }
}

with open("model_artifacts.json", "w") as f:
    json.dump(artifacts, f, indent=2)

print("Exporting trained XGBoost pipeline to trained_xgb_pipeline.joblib...")
joblib.dump(xgb_model, "trained_xgb_pipeline.joblib")
joblib.dump(lr_model, "trained_lr_pipeline.joblib")
print("Done! Artifacts saved successfully.")

