# api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from predictor import generate_predictions

# Create the FastAPI application
app = FastAPI(
    title="Restaurant Order Prediction API",
    description="An API to predict order probabilities for menu items based on time, weather, and events.",
    version="1.0.0"
)

# Define the structure of the input data for the API request
# This provides automatic data validation
class PredictionInput(BaseModel):
    target_date: str = Field(..., example="2024-12-25", description="Date for prediction in YYYY-MM-DD format.")
    target_hour: int = Field(..., ge=0, le=23, example=19, description="Hour of the day (0-23).")
    is_special_event: bool = Field(..., example=False, description="True if there is a special local event.")

# Define the API endpoint
@app.post("/predict", tags=["Predictions"])
def create_prediction(payload: PredictionInput):
    """
    Receives prediction inputs, calls the prediction engine, and returns results.
    """
    # Call the core logic from predictor.py
    prediction_result = generate_predictions(
        target_date_str=payload.target_date,
        target_hour=payload.target_hour,
        is_special_event=payload.is_special_event
    )

    # If the predictor returned an error, raise an HTTP Exception
    if "error" in prediction_result:
        raise HTTPException(status_code=400, detail=prediction_result["error"])

    # Otherwise, return the successful prediction
    return prediction_result

# A simple root endpoint to confirm the API is running
@app.get("/", tags=["Health Check"])
def read_root():
    return {"status": "ok", "message": "Prediction API is running. Go to /docs to see the API documentation."}
