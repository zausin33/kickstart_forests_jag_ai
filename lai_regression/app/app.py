import pickle
import os

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, create_model
from lai_regression.src import data

script_path = os.path.dirname(os.path.abspath(__file__))

app = FastAPI()
default_model_path = os.path.join(script_path, "..", "results", "models", "regression", "full", "best_model.pickle")
model_path = os.environ.get("MODEL_PATH", default_model_path)

with open(model_path, 'rb') as f:
    model = pickle.load(f)


# Use create_model to dynamically create the Pydantic model
PredictionInput = create_model(
    'PredictionInput',
    __base__=BaseModel,
    **{name: (float, ...) for name in data.COLS_NUMERICAL},
    **{name: (str, ...) for name in data.COLS_CATEGORICAL},
)

@app.get("/model_info/")
def model_info():
    return str(model)


@app.post("/predict/")
def predict(input_data: PredictionInput):
    data = pd.DataFrame([dict(input_data)])
    prediction = model.predict(data)
    return prediction
