import pandas as pd
from sklearn.ensemble import IsolationForest
import os
import pickle

def train_anomaly_detector(data_path="data/processed/er_ts_data_processed.parquet", model_dir="backend/models"):
    """
    Trains an Isolation Forest to detect anomalous peaks in arrivals or wait times.
    """
    print("Loading data for Anomaly Detection...")
    df = pd.read_parquet(data_path)
    
    # We use basic high-level stats for anomaly detection
    features = ['arrival_count', 'occupancy_at_arrival', 'wait_time_mins']
    
    X = df[features].dropna()
    
    # Assuming about 5% of data are true anomalies (extreme spikes)
    contamination = 0.05
    
    print("Training Isolation Forest Anomaly Detector...")
    iso_forest = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
    iso_forest.fit(X)
    
    # Sanity check
    preds = iso_forest.predict(X)
    anomalies = (preds == -1).sum()
    print(f"Detected {anomalies} anomalies out of {len(X)} records ({anomalies/len(X)*100:.2f}%)")
    
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "iso_forest_anomaly.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(iso_forest, f)
        
    print(f"Anomaly Detection model saved to {model_path}")

if __name__ == "__main__":
    train_anomaly_detector()
