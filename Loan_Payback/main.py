import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import pandas as pd
import io
import uvicorn
from prediction import run_pipeline, load_artifacts
from pydantic import BaseModel,Field
from typing import List, Optional

# Initialize FastAPI app
app = FastAPI(
    title="Loan Payback Predictor API",
    description="API for predicting loan payback probability using a trained ML model",
    version="1.0.0"
)


# ============================================================
# Pydantic Models
# ============================================================
class LoanPredictionInput(BaseModel):
    """Single loan record for prediction"""
    credit_score: float = Field(..., example="694")
    annual_income: float = Field(..., example="22108.02")
    loan_amount: float= Field(..., example="4593.10")
    interest_rate: float= Field(..., example="12.92")
    debt_to_income_ratio: float= Field(..., example="0.166")


class LoanPredictionResponse(BaseModel):
    """Response for single loan prediction"""
    probability: float
    predicted_class: int
    message: str


class BatchPredictionResponse(BaseModel):
    """Response metadata for batch predictions"""
    total_records: int
    message: str
    output_file: str


# ============================================================
# Health Check Endpoint
# ============================================================
@app.get("/health")
def health_check():
    """Check API health and model availability"""
    try:
        load_artifacts()
        return {
            "status": "healthy",
            "message": "API is running and model artifacts are loaded"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")


# ============================================================
# Single Prediction Endpoint
# ============================================================
@app.post("/predict/single", response_model=LoanPredictionResponse)
def predict_single(loan: LoanPredictionInput):
    """
    Predict loan payback for a single record.
    
    Args:
        loan: LoanPredictionInput with loan details
        
    Returns:
        LoanPredictionResponse with probability and predicted class
    """
    try:
        # Convert input to DataFrame
        df = pd.DataFrame([loan.model_dump()])
        
        # Run prediction pipeline
        prob, pred_class = run_pipeline(df)
        
        return LoanPredictionResponse(
            probability=float(prob[0]),
            predicted_class=int(pred_class[0]),
            message="Prediction successful"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


# ============================================================
# Batch Prediction Endpoint (CSV Upload)
# ============================================================
@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    """
    Predict loan payback for multiple records via CSV upload.
    
    Args:
        file: CSV file with loan records
        
    Returns:
        CSV file with predictions appended
    """
    try:
        # Read uploaded CSV
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file is empty")
        
        # Run prediction pipeline
        prob, pred_class = run_pipeline(df)
        
        # Add predictions to dataframe
        df["probability"] = prob
        df["predicted_class"] = pred_class
        
        # Save to temporary file
        os.makedirs("temp_outputs", exist_ok=True)
        output_path = "temp_outputs/batch_predictions.csv"
        df.to_csv(output_path, index=False)
        
        return FileResponse(
            path=output_path,
            filename="loan_predictions.csv",
            media_type="text/csv"
        )
    except pd.errors.ParserError:
        raise HTTPException(status_code=400, detail="Invalid CSV format")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction failed: {str(e)}")


# ============================================================
# Batch Prediction Endpoint (JSON)
# ============================================================
@app.post("/predict/batch-json")
def predict_batch_json(loans: List[LoanPredictionInput]):
    """
    Predict loan payback for multiple records via JSON.
    
    Args:
        loans: List of LoanPredictionInput objects
        
    Returns:
        List of predictions with probabilities and classes
    """
    try:
        if not loans:
            raise HTTPException(status_code=400, detail="No loans provided")
        
        # Convert to DataFrame
        df = pd.DataFrame([loan.model_dump() for loan in loans])
        
        # Run prediction pipeline
        prob, pred_class = run_pipeline(df)
        
        # Build response
        results = [
            {
                "index": i,
                "probability": float(prob[i]),
                "predicted_class": int(pred_class[i])
            }
            for i in range(len(prob))
        ]
        
        return {
            "total_records": len(results),
            "predictions": results,
            "message": "Batch prediction successful"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction failed: {str(e)}")


# ============================================================
# Model Info Endpoint
# ============================================================
@app.get("/model/info")
def get_model_info():
    """Get information about the loaded model and features"""
    try:
        preprocessor, selector, model, selected_features, expected_columns = load_artifacts()
        
        return {
            "expected_input_columns": expected_columns,
            "selected_features": selected_features.tolist() if hasattr(selected_features, 'tolist') else selected_features,
            "model_type": str(type(model).__name__),
            "n_selected_features": len(selected_features) if hasattr(selected_features, '__len__') else None,
            "message": "Model information retrieved successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve model info: {str(e)}")


# ============================================================
# Root Endpoint
# ============================================================
@app.get("/")
def root():
    """Welcome endpoint"""
    return {
        "message": "Welcome to Loan Payback Predictor API",
        "endpoints": {
            "health": "GET /health",
            "single_prediction": "POST /predict/single",
            "batch_prediction_csv": "POST /predict/batch",
            "batch_prediction_json": "POST /predict/batch-json",
            "model_info": "GET /model/info",
            "docs": "/docs"
        }
    }


# ============================================================
# Run Server
# ============================================================
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)