from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
import pickle
import pandas as pd

# ================================
# 1. Load Models & Data
# ================================
try:
    with open("Arima.pkl", "rb") as f:
        models = pickle.load(f)
    print("✅ Models loaded successfully.")
except FileNotFoundError:
    print("❌ Model file not found.")
    exit()

try:
    df = pd.read_csv("Drug_Utilization_data(2015-2024).csv")
    print("✅ Historical data loaded successfully.")
except FileNotFoundError:
    print("❌ Historical data file not found.")
    exit()

# ================================
# 2. FastAPI Setup
# ================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI on PythonAnywhere!"}
# ================================
# 3. Pydantic Schema
# ================================
class ForecastRequest(BaseModel):
    drug_name: str
    steps: int

# ================================
# 4. Forecast Logic
# ================================
def forecast_drug_json(drug_name: str, steps: int = 5):
    if drug_name not in models:
        return {"error": f"No model found for {drug_name}"}
    
    hist = df[df["Gnrc_Name"] == drug_name].sort_values("Year")
    if hist.empty:
        return {"error": f"No historical data found for {drug_name}"}

    last_year = hist["Year"].max()
    future_years = list(range(last_year + 1, last_year + 1 + steps))
    
    forecast_data = {}
    for target in ["Total_Claims", "Total_Beneficiaries"]:
        if target not in models[drug_name]:
            forecast_data[target] = [0] * steps
            continue
        fitted = models[drug_name][target]
        forecast = fitted.forecast(steps=steps)
        forecast_data[target] = [int(x) for x in forecast]

    return {
        "drug": drug_name,
        "forecast_years": future_years,
        "historical": {
            "years": hist["Year"].tolist(),
            "Total_Claims": hist["Total_Claims"].astype(int).tolist(),
            "Total_Beneficiaries": hist["Total_Beneficiaries"].astype(int).tolist(),
        },
        "forecast": forecast_data,
    }


@app.post("/api/drug-utilization-forecast")
async def drug_utilization_forecast(request: ForecastRequest):
    data = forecast_drug_json(request.drug_name, request.steps)
    print(data)
    if "error" in data:
        raise HTTPException(status_code=404, detail=data["error"])
    return data

# (Keep your GET if needed)
@app.get("/forecast")
async def get_forecast(drug_name: str, steps: int = 5):
    data = forecast_drug_json(drug_name, steps)
    if "error" in data:
        raise HTTPException(status_code=404, detail=data["error"])
    return data
