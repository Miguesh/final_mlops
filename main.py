from typing import List
from pandas import DataFrame
import numpy as np
import pandas as pd
from joblib import load
from pydantic import BaseModel, ValidationError
from fastapi import FastAPI, HTTPException


model = load('pkl/model.pkl')



app = FastAPI(
    title="Salary Prediction API",
    description="API para predicción de salario (regresión)",
    version="1.0.0"
)

class DataPredict(BaseModel):
    data_to_predict: list[list] = [[1.0905917949529544,0,0,0,1,0,0,1,0,0,1,0],[1.0905917949529544,1,0,0,1,0,0,1,0,0,1,0]]


@app.get("/")
def home():
    return {
        "service": "Salary Prediction API",
        "status": "running"
    }


@app.post("/predict")
def predict(request: DataPredict):
    try:
        data = request.data_to_predict
        df_data = DataFrame(data,columns=["num__remote_ratio", "cat__experience_level_EN", "cat__experience_level_EX", "cat__experience_level_MI", "cat__experience_level_SE", "cat__employment_type_CT", "cat__employment_type_FL", "cat__employment_type_FT", "cat__employment_type_PT", "cat__company_size_L", "cat__company_size_M", "cat__company_size_S"])

        if len(data) == 0:
            raise ValueError("data_to_predict no puede estar vacío")

        predictions = model.predict(df_data)


        return {
            "n_predictions": len(predictions),
            "predictions": predictions.tolist()
        }
    
        

    except ValidationError as ve:
        raise HTTPException(status_code=400, detail=ve.errors())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
