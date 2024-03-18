import pickle
import os

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

script_path = os.path.dirname(os.path.abspath(__file__))

app = FastAPI()
default_model_path = os.path.join(script_path, "..", "results", "models", "regression", "full", "best_model.pickle")
model_path = os.environ.get("MODEL_PATH", default_model_path)

with open(model_path, 'rb') as f:
    model = pickle.load(f)


class PredictionInput(BaseModel):
    id: int
    lai: float
    wetness: float
    treeSpecies: str
    """Sentinel_2A_492.4: float
    Sentinel_2A_559.8: float
    Sentinel_2A_664.6: float
    Sentinel_2A_704.1: float
    Sentinel_2A_740.5: float
    Sentinel_2A_782.8: float
    Sentinel_2A_832.8: float
    Sentinel_2A_864.7: float
    Sentinel_2A_1613.7: float
    Sentinel_2A_2202.4: float"""


@app.get("/model_info/")
def model_info():
    return str(model)


@app.post("/predict/")
def predict(input_data: PredictionInput):
    data = pd.DataFrame([dict(input_data)])
    prediction = model.predict(data)
    return prediction
