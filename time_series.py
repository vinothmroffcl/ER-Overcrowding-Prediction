import pandas as pd
import numpy as np
from prophet import Prophet
import os
import pickle

def train_prophet(data_path="data/processed/er_ts_data_processed.parquet", model_dir="backend/models"):
    print("Loading time-series data for Prophet...")
    df = pd.read_parquet(data_path)
    
    # Prophet requires 'ds' (datetime target) and 'y' (numeric target)
    prophet_df = df.reset_index()[['arrival_time', 'arrival_count']].rename(
        columns={'arrival_time': 'ds', 'arrival_count': 'y'}
    )
    
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
    m.add_country_holidays(country_name='US')
    
    print("Training Prophet Model...")
    m.fit(prophet_df)
    
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "prophet_arrivals.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(m, f)
        
    print(f"Prophet Model saved to {model_path}")

def predict_future(model_path="backend/models/prophet_arrivals.pkl", periods=24):
    with open(model_path, "rb") as f:
        m = pickle.load(f)
        
    # 'h' for hourly frequency
    future = m.make_future_dataframe(periods=periods, freq='h')
    forecast = m.predict(future)
    
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)

if __name__ == "__main__":
    train_prophet()
