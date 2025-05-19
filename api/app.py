import mlflow
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from contextlib import asynccontextmanager

# Features from your params.yaml
NUMERIC_FEATURES = ["Age", "SibSp", "Parch", "Fare"]
CATEGORICAL_FEATURES = ["Pclass", "Sex", "Embarked"]

# Define input data model
class ModelInput(BaseModel):
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Pclass: int
    Sex: str
    Embarked: str

    class Config:
        json_schema_extra = {
            "example": {
                "Age": 29.0,
                "SibSp": 0,
                "Parch": 0,
                "Fare": 7.25,
                "Pclass": 3,
                "Sex": "male",
                "Embarked": "S"
            }
        }

# Use FastAPI lifespan for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        # print("Connecting to MLflow...")
        # mlflow.set_tracking_uri("http://localhost:5000")
        # print("Connected to MLflow. Loading model...")
        # model_name = "random_forest_model"
        # model_version = 1
        # model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_version}")
        # print("Model loaded successfully.")
        import joblib
        model_path = "../models/random_forest_model.joblib"
        model = joblib.load(model_path)
        yield
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

# >>> Create the app BEFORE defining endpoints <<<
app = FastAPI(title="Titanic Random Forest Model API", lifespan=lifespan)

@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/predict")
async def predict(input_data: ModelInput):
    try:
        # Convert input to DataFrame
        input_dict = input_data.dict()
        input_df = pd.DataFrame([input_dict])

        # One-hot encode categorical features to match training
        input_df = pd.get_dummies(input_df, columns=CATEGORICAL_FEATURES)
        # Align columns with model's expected input
        model_cols = model.metadata.get_input_schema().input_names() if hasattr(model, "metadata") else input_df.columns
        input_df = input_df.reindex(columns=model_cols, fill_value=0)

        prediction = model.predict(input_df)
        return {
            "prediction": prediction.tolist(),
            "input_data": input_dict
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)