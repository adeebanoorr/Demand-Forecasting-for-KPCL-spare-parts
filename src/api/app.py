from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os

app = FastAPI(title="Spare Parts Forecast API")

# Allow frontend (Lovable site) to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_PATH = "data/processed"

# -----------------------------
# Helper Functions
# -----------------------------

def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path).to_dict(orient="records")
    return []

# -----------------------------
# API Endpoints
# -----------------------------

@app.get("/")
def home():
    return {"message": "Spare Parts Forecast API Running"}

@app.get("/items")
def get_items():
    return {
        "items": [
            "082.03.110.50.",
            "082.04.030.50.",
            "082.08.000.50.",
            "084.19.001.50.",
            "085.00.003.50.",
            "336.40.401.50.",
            "351.03.301.50.",
            "993.00.311.00.",
        ]
    }

@app.get("/forecast/{item_code}")
def get_forecast(item_code: str):
    path = f"{BASE_PATH}/final_forecasts/{item_code}_forecast.csv"
    return {"data": load_csv(path)}

@app.get("/validation/{item_code}")
def get_validation(item_code: str):
    path = f"{BASE_PATH}/validation/{item_code}_validation.csv"
    return {"data": load_csv(path)}

@app.get("/model-comparison/{item_code}")
def get_model_comparison(item_code: str):
    path = f"{BASE_PATH}/model_comparison/{item_code}_comparison.csv"
    return {"data": load_csv(path)}