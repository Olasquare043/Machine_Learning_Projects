# main.py
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from prediction import run_pipeline, feature_engineering, load_artifacts
import io

app = FastAPI(title="Loan Payback Prediction API")

# Load artifacts once at startup
preprocessor, selector, model, selected_features, expected_input_columns = load_artifacts()


# -------------------------------------------------------
# 1) SINGLE RECORD PREDICTION (JSON)
# -------------------------------------------------------
class SingleRecord(BaseModel):
    gender: str
    marital_status: str
    employment_status: str
    loan_purpose: str
    education_level: str
    grade_subgrade: str
    loan_amount: float
    interest_rate: float
    annual_income: float
    credit_score: float
    debt_to_income_ratio: float


@app.post("/predict")
def predict_single(record: SingleRecord):
    try:
        # Convert record → DataFrame
        df = pd.DataFrame([record.dict()])

        # Run the ML pipeline
        prob, pred_class = run_pipeline(df)

        return {
            "probability": float(prob[0]),
            "predicted_class": int(pred_class[0])
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# -------------------------------------------------------
# 2) CSV UPLOAD PREDICTION → KAGGLE SUBMISSION FORMAT
# -------------------------------------------------------
@app.post("/predict-csv")
def predict_csv(file: UploadFile = File(...)):
    try:
        # READ CSV INTO DATAFRAME
        contents = file.file.read()
        df = pd.read_csv(io.BytesIO(contents))

        if "id" not in df.columns:
            raise HTTPException(400, detail="CSV must contain an 'id' column.")

        ids = df["id"]
        df_no_id = df.drop(columns=["id"])

        # RUN PIPELINE
        prob, pred_class = run_pipeline(df_no_id)

        # BUILD KAGGLE SUBMISSION FORMAT
        submission = pd.DataFrame({
            "id": ids,
            "loan_paid_back": prob
        })

        return submission.to_dict(orient="records")

    except Exception as e:
        raise HTTPException(400, detail=str(e))
