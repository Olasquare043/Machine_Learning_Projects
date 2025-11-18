# main.py
from fastapi.responses import FileResponse
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from prediction import run_pipeline, feature_engineering, load_artifacts
import io
import os

# Initialize FastAPI app
app = FastAPI(
    title="Loan Payback Predictor API",
    description="API for predicting loan payback probability",
    version="1.0.0"
)

# Load artifacts once at startup
preprocessor, selector, model, selected_features, expected_input_columns = load_artifacts()


# -------------------------------------------------------
# SINGLE RECORD PREDICTION (JSON)
# -------------------------------------------------------
class SingleRecord(BaseModel):
    gender: str =Field(..., example="Male")
    marital_status: str =Field(..., example="Single")
    employment_status: str =Field(..., example="Employed")
    loan_purpose: str =Field(..., example="Debt Consolidation")
    education_level: str =Field(..., example=r"Bachelor's")
    grade_subgrade: str =Field(..., example="C2")
    loan_amount: float =Field(..., example=15000)
    interest_rate: float =Field(..., example=0.12)
    annual_income: float =Field(..., example=48000)
    credit_score: float =Field(..., example=680)
    debt_to_income_ratio: float =Field(..., example=0.28)

# root route
@app.get("/")
def root():
    """WELCOME """
    return {"message":"Welcome to loan payback predicting App"}

# prediction route
@app.post("/predict")
def predict_single(record: SingleRecord):
    try:
        # Convert record → DataFrame
        df = pd.DataFrame([record.model_dump()])

        # Run the ML pipeline
        prob, pred_class = run_pipeline(df)

        return {
            "probability": float(prob[0]),
            "predicted_class": int(pred_class[0]),
            "Interpretation": "The customer will not likely payback the loan" if int(pred_class[0])== 0 else "The customer will likely payback"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# -------------------------------------------------------
# CSV UPLOAD PREDICTION → KAGGLE SUBMISSION FORMAT
# -------------------------------------------------------
@app.post("/predict-csv")
async def predict_csv(file: UploadFile = File(...)):
    """
    Predict loan payback for multiple records via CSV upload.
    Args: file: CSV file with loan records
    Returns: CSV file with predictions appended
    """
    try:
        # READ CSV INTO DATAFRAME
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        if df.empty:
            raise HTTPException (status_code=400, detail="CSV file is empty")
        
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

        # SAVE TO TEMPORY FILE
        os.makedirs("temp_outputs",exist_ok=True)
        output_path="temp_outputs/batch_predictions.csv"
        submission.to_csv(output_path,index=False)

        return FileResponse (
            path=output_path,
            filename="Loan_predictions.csv",
            media_type="text/csv")
    except Exception as e:
        raise HTTPException(400, detail=str(e))
