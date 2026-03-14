import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import os
import pickle

def train_overcrowding_classifier(data_path="data/processed/er_ts_data_processed.parquet", model_dir="backend/models"):
    """
    Trains an XGBoost classifier to predict the likelihood of an overcrowding state
    (Green, Yellow, Red) for the next hour based on current hour metrics.
    """
    print("Loading data for Overcrowding Classification...")
    df = pd.read_parquet(data_path)
    
    # We want to predict if the NEXT hour will be overcrowded.
    # We will build a simple rule for overcrowding status (0: Green, 1: Yellow, 2: Red)
    # based on the next hour's target wait time & occupancy
    df['target_occupancy_next_1h'] = df['occupancy_at_arrival'].shift(-1)
    df['target_wait_time_next_1h'] = df['wait_time_mins'].shift(-1)
    df = df.dropna()
    
    def get_status(row):
        obs = row['target_occupancy_next_1h']
        wait = row['target_wait_time_next_1h']
        if obs > 80 or wait > 180: return 2 # Red
        if obs > 50 or wait > 60: return 1  # Yellow
        return 0 # Green
        
    df['target_status'] = df.apply(get_status, axis=1)
    
    features = ['arrival_count', 'occupancy_at_arrival', 'wait_time_mins', 
                'hour', 'day_of_week', 'month', 'is_holiday', 'is_flu_season']
    weather_cols = [c for c in df.columns if 'weather_' in c]
    features.extend(weather_cols)
    
    X = df[features]
    y = df['target_status']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Check class distribution
    print(f"Class distribution in train:\n{y_train.value_counts(normalize=True)}")
    
    print("Training XGBoost Classifier...")
    model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, 
                              random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    print(f"Classification Accuracy: {accuracy_score(y_test, preds):.2f}")
    print("Classification Report:")
    print(classification_report(y_test, preds, zero_division=0))
    
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "xgb_overcrowding.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
        
    print(f"Classification Model saved to {model_path}")

if __name__ == "__main__":
    train_overcrowding_classifier()
