This project is an end-to-end Machine Learning based system that predicts:

- Actual mobile phone temperature (°C) using regression
- Overheating risk (Yes/No) using classification
- Actionable user advice to reduce overheating

The system combines ML models + real-world decision logic and is deployed using a Streamlit web interface.

Problem Statement

Mobile phones often overheat due to:

High CPU/GPU usage (gaming, video processing)

Charging during heavy usage

High ambient temperature

Poor ventilation (phone case on)

Overheating affects:

Performance

Battery health

User safety

This project aims to predict overheating early and provide preventive recommendations.

Solution Approach

The system uses two ML models:

Temperature Prediction (Regression)

Model: Random Forest Regressor

Output: Predicted phone temperature in °C

2️⃣ Overheat Risk Prediction (Classification)

Model: Random Forest Classifier

Output: Probability of overheating

3️⃣ Final Decision Logic

If predicted temperature ≥ 50°C 
OR
If overheating probability ≥ 0.4
→ Raise OVERHEATING ALERT


This logic reduces false negatives and improves safety.

🧪 Features Used

Ambient temperature

CPU usage

GPU usage

RAM usage

Screen brightness

Battery level

Charging status

Phone case status

Background apps count

Time since boot

App type (idle / social / video / game)

Network type (WiFi / 4G / 5G)

Engineered features (mean load, interactions, logs)

⚙️ Tech Stack

Python

scikit-learn

pandas / numpy

Streamlit

joblib

📂 Project Structure
overheat/
│
├── app.py                     # Streamlit app
├── temperature_regressor.pkl  # Regression model
├── overheat_classifier.pkl    # Classification model
├── model_features.pkl         # Saved feature schema
├── requirements.txt
├── README.md

▶️ How to Run the Project
1️⃣ Create & activate virtual environment
python -m venv mlenv
mlenv\Scripts\activate

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run Streamlit app
streamlit run app.py


App will open at:

http://localhost:8501

📊 Output Example

The app displays:

✅ Predicted Temperature (°C)

📊 Overheat Probability

🚨 Final Alert (SAFE / OVERHEATING RISK)

💡 Suggested Action (e.g., unplug charger, close apps)

💡 Sample Advice Logic

High GPU usage → Close heavy gaming apps

Charging during load → Unplug charger

Phone case on → Remove case temporarily

Otherwise → Reduce brightness / rest device

🏆 Key Highlights

✔ Regression + Classification combined

✔ Handles class imbalance

✔ Feature mismatch safely handled

✔ Models saved & reused using .pkl

✔ Real-time interactive interface

✔ Industry-style ML workflow

📈 Future Improvements

Time-series temperature prediction (LSTM)

Personalized models per device

On-device inference (TensorFlow Lite)

Cloud deployment (Render / HuggingFace Spaces)

Real sensor data integration

👤 Author

Gaurav Chauhan
B.Tech Graduate