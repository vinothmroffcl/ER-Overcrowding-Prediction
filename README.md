# Emergency Room (ER) Overcrowding Prediction System

A complete, professional Data Science and Software Engineering project for predicting ER patient traffic, wait times, and capacity anomalies.

## 🚀 Features
- **Real-Time ER Dashboard**: A beautiful, glassmorphic UI built with Vanilla JS, Chart.js, and TailwindCSS.
- **Machine Learning Engine**: 
  - **Prophet**: Forecasts hourly patient arrivals for the next 24 hours.
  - **Random Forest**: Predicts resource demand (Beds, Doctors, Nurses) for the next hour.
  - **XGBoost Classifier**: Classifies ER status into Green, Yellow, or Red based on historical and real-time wait times and occupancy.
  - **Isolation Forest**: Scans continuous real-time streams to detect anomalous patient surges.
- **Real-Time Data Pipeline**: Apache Kafka streams simulated patient arrival data, which is consumed and cached in Redis for blazing-fast API reads.
- **Fully Dockerized**: The entire architecture stands up with a single `docker-compose` command.

## 📦 Project Structure
- `backend/`: FastAPI application, Machine Learning scripts, and Kafka Producer/Consumer.
- `backend/ml_models/`: Training pipelines for the ML algorithms.
- `backend/data_pipeline/`: Scripts to generate years of synthetic ER data and engineered features.
- `frontend/`: Real-time interactive UI running directly on the blazing-fast API static mount.

## ⚡ How to Run

### Option 1: Full Docker Architecture (Recommended)
This spins up Zookeeper, Kafka, Redis, the FastAPI Backend, and the Real-time Producer/Consumer scripts.

1. Install Docker Desktop.
2. Run the following command in the root directory:
```bash
docker-compose up --build
```
3. Open your browser and navigate to: **http://localhost:8000**

### Option 2: Local Python Execution (No Docker needed)
The API has a built-in fallback: if Kafka/Redis are offline, it will simulate a dynamic state internally so you can still view the dashboard and ML predictions!

1. Create a virtual environment and activate it:
```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```
2. Install dependencies:
```bash
cd backend
pip install -r requirements.txt
```
3. Generate data and train the ML models (Optional: pre-trained models are likely already inside `backend/models`):
```bash
python data_pipeline/generate_dataset.py
python data_pipeline/preprocess.py
python ml_models/time_series.py
python ml_models/regression.py
python ml_models/classification.py
python ml_models/anomaly_detection.py
```
4. Start the FastAPI Server:
```bash
uvicorn main:app --reload --port 8000
```
5. Open your browser and navigate to: **http://127.0.0.1:8000**
