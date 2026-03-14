import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import pickle

def train_resource_demand_model(data_path="data/processed/er_ts_data_processed.parquet", model_dir="backend/models"):
    """
    Trains a model to predict the resource demand (e.g. occupancy and wait time).
    """
    print("Loading data for Regression (Resource Demand)...")
    df = pd.read_parquet(data_path)
    
    # We will predict next hour's occupancy based on current hour data
    df['target_occupancy_next_1h'] = df['occupancy_at_arrival'].shift(-1)
    df = df.dropna()
    
    # Features for resource demand
    features = ['arrival_count', 'occupancy_at_arrival', 'wait_time_mins', 
                'hour', 'day_of_week', 'month', 'is_holiday', 'is_flu_season']
    weather_cols = [c for c in df.columns if 'weather_' in c]
    features.extend(weather_cols)
    
    X = df[features]
    y = df['target_occupancy_next_1h']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    print("Training Random Forest Regressor for Next-Hour Occupancy...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    preds = rf.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    
    print(f"Regression Test MSE: {mse:.2f}")
    print(f"Regression Test MAE: {mae:.2f}")
    
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "rf_occupancy_demand.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(rf, f)
        
    print(f"Regression Model saved to {model_path}")

if __name__ == "__main__":
    train_resource_demand_model()
