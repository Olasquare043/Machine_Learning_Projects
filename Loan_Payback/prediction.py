# prediction.py
import os
import joblib
import pandas as pd
import numpy as np

# metrics
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# ----------------------------------------------------
# PATHS
# ----------------------------------------------------
def load_artifacts():
    ARTIFACTS_DIR = "artifacts"
    PREPROCESSOR_PATH = os.path.join(ARTIFACTS_DIR, "preprocessor.pkl")
    SELECTOR_PATH = os.path.join(ARTIFACTS_DIR, "selector.pkl")
    MODEL_PATH = os.path.join(ARTIFACTS_DIR, "best_boosting_model_smot.pkl")
    FEATURE_NAMES_PATH = os.path.join(ARTIFACTS_DIR, "feature_names.pkl")

    # ----------------------------------------------------
    # Load Artifacts
    # ----------------------------------------------------
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    selector = joblib.load(SELECTOR_PATH)
    model = joblib.load(MODEL_PATH)
    selected_features = joblib.load(FEATURE_NAMES_PATH)

    # Identify columns expected by preprocessor
    expected_input_columns = (
        preprocessor.feature_names_in_.tolist()
        if hasattr(preprocessor, "feature_names_in_")
        else None
    )
    return preprocessor, selector, model, selected_features,expected_input_columns
# ----------------------------------------------------
# Feature Engineering
# ----------------------------------------------------
def feature_engineering(df):
    df = df.copy()

    if 'interest_rate' in df.columns and 'debt_to_income_ratio' in df.columns:
        df['interest_burden'] = df['interest_rate'] * df['debt_to_income_ratio']

    if 'loan_amount' in df.columns and 'annual_income' in df.columns:
        df['loan_income_ratio'] = df['loan_amount'] / df['annual_income']

    if 'credit_score' in df.columns and 'annual_income' in df.columns:
        df['credit_efficiency'] = df['credit_score'] / (df['annual_income'] / 1000 + 1)

    if 'annual_income' in df.columns:
        df['monthly_income'] = df['annual_income'] / 12

    if 'loan_amount' in df.columns and 'credit_score' in df.columns:
        df['loan_to_credit'] = df['loan_amount'] / (df['credit_score'] + 1)

    if 'loan_amount' in df.columns and 'interest_rate' in df.columns:
        df['risk_weighted_amount'] = df['loan_amount'] * df['interest_rate']

    return df


# ----------------------------------------------------
# Apply preprocessing → selection → prediction
# ----------------------------------------------------
def run_pipeline(df):
    df = df.copy()
    preprocessor, selector, model, selected_features,expected_input_columns= load_artifacts()
    # Feature engineering
    df = feature_engineering(df)

    # Ensure expected columns exist
    if expected_input_columns:
        missing = [c for c in expected_input_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        df = df[expected_input_columns]

    # Preprocessing
    X_processed = preprocessor.transform(df)

    # Feature selection
    X_selected = selector.transform(X_processed)

    # Predict proba & class
    prob = model.predict_proba(X_selected)[:, 1]
    pred_class = (prob >= 0.7).astype(int) # Using 0.70 threshold

    return prob, pred_class


# ----------------------------------------------------
# Kaggle submission function
# ----------------------------------------------------
def predict_kaggle(test_csv_path, output_path="submissions/loan_payback_predictions.csv"):
    os.makedirs("submissions", exist_ok=True)

    test_df = pd.read_csv(test_csv_path)
    ids = test_df["id"]

    prob, pred_class = run_pipeline(test_df.drop(columns=["id"]))

    submission = pd.DataFrame({
        "id": ids,
        "loan_paid_back": prob
    })

    submission.to_csv(output_path, index=False)
    print(f"Kaggle submission saved → {output_path}")


# ----------------------------------------------------
# General batch prediction from CSV
# ----------------------------------------------------
def predict_csv(input_path, output_path):
    df = pd.read_csv(input_path)
    prob, pred_class = run_pipeline(df)

    result = df.copy()
    result["probability"] = prob
    result["predicted_class"] = pred_class

    result.to_csv(output_path, index=False)
    print(f"Batch prediction saved → {output_path}")
    
# ----------------------------------------------------
# Evaluation Metrics 
# ----------------------------------------------------
def model_evaluation(df,true_labels):
    prob, pred_class=run_pipeline(df)
    
    # convert label to numpy
    y_true= np.array(true_labels)

    # checking the ROC-AUC Probability of class 1 (repay)
    roc_auc= roc_auc_score(y_true,prob)

    # checking recall for Default (class 0)
    recall_default= recall_score(y_true,pred_class,pos_label=0)

    # checking precision for Repay (class 1)
    precision_repay= precision_score(y_true,pred_class,pos_label=1)

    metrics= {
    "roc_auc": round(roc_auc,2),
    "recall_default": round(recall_default,2),
    "precision_repay": round(precision_repay,2)
    }
    return metrics
