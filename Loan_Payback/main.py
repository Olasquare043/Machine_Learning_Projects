# main.py
<<<<<<< HEAD
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
=======
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


>>>>>>> 545695f403a8573ea25d8cae6f2063f3f58327e1
@app.post("/predict")
def predict_single(record: SingleRecord):
    try:
        # Convert record → DataFrame
<<<<<<< HEAD
        df = pd.DataFrame([record.model_dump()])
=======
        df = pd.DataFrame([record.dict()])
>>>>>>> 545695f403a8573ea25d8cae6f2063f3f58327e1

        # Run the ML pipeline
        prob, pred_class = run_pipeline(df)

        return {
            "probability": float(prob[0]),
<<<<<<< HEAD
            "predicted_class": int(pred_class[0]),
            "Interpretation": "The customer will not likely payback the loan" if int(pred_class[0])== 0 else "The customer will likely payback"
=======
            "predicted_class": int(pred_class[0])
>>>>>>> 545695f403a8573ea25d8cae6f2063f3f58327e1
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# -------------------------------------------------------
<<<<<<< HEAD
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
        
=======
# 2) CSV UPLOAD PREDICTION → KAGGLE SUBMISSION FORMAT
# -------------------------------------------------------
@app.post("/predict-csv")
def predict_csv(file: UploadFile = File(...)):
    try:
        # READ CSV INTO DATAFRAME
        contents = file.file.read()
        df = pd.read_csv(io.BytesIO(contents))

>>>>>>> 545695f403a8573ea25d8cae6f2063f3f58327e1
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

<<<<<<< HEAD
        # SAVE TO TEMPORY FILE
        os.makedirs("temp_outputs",exist_ok=True)
        output_path="temp_outputs/batch_predictions.csv"
        submission.to_csv(output_path,index=False)

        return FileResponse (
            path=output_path,
            filename="Loan_predictions.csv",
            media_type="text/csv")
=======
        return submission.to_dict(orient="records")

>>>>>>> 545695f403a8573ea25d8cae6f2063f3f58327e1
    except Exception as e:
        raise HTTPException(400, detail=str(e))
