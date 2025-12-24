Mobile Phone Overheating Prediction System
Project Overview

This project is an end-to-end Machine Learning–based system designed to predict mobile phone overheating and provide preventive recommendations. The system performs three main tasks:

Predicts the actual mobile phone temperature (°C) using a regression model

Predicts overheating risk (Yes/No) using a classification model

Provides actionable user advice to reduce overheating

The solution combines machine learning models with real-world decision logic and is deployed using a Streamlit-based web interface.

Problem Statement

Mobile phones often overheat due to multiple factors such as:

High CPU and GPU usage during gaming or video processing

Charging the device during heavy usage

High ambient environmental temperature

Poor ventilation, for example when a phone case is used

Overheating negatively impacts:

Device performance

Battery health and lifespan

User safety

This project aims to predict overheating at an early stage and suggest preventive actions to avoid potential damage.

Solution Approach

The system is built using two machine learning models:

1. Temperature Prediction (Regression)

Model: Random Forest Regressor

Output: Predicted mobile phone temperature in degrees Celsius

2. Overheat Risk Prediction (Classification)

Model: Random Forest Classifier

Output: Probability of overheating

3. Final Decision Logic

An overheating alert is raised if:

The predicted temperature is greater than or equal to 50°C
OR

The predicted overheating probability is greater than or equal to 0.4

This combined logic helps reduce false negatives and improves user safety.

Features Used

The model uses the following input features:

Ambient temperature

CPU usage

GPU usage

RAM usage

Screen brightness

Battery level

Charging status

Phone case status

Number of background applications

Time since device boot

Application type (idle, social, video, game)

Network type (WiFi, 4G, 5G)

Additional engineered features include mean load values, interaction features, and logarithmic transformations.

Tech Stack

Python

scikit-learn

pandas

numpy

Streamlit

joblib

How to Run the Project
Step 1: Create and activate a virtual environment
python -m venv mlenv
mlenv\Scripts\activate

Step 2: Install required dependencies
pip install -r requirements.txt

Step 3: Run the Streamlit application
streamlit run app.py


The application will be accessible at:

http://localhost:8501

Output Description

The Streamlit application displays:

Predicted mobile phone temperature (°C)

Overheating probability

Final system alert (SAFE or OVERHEATING RISK)

Suggested action to reduce overheating

Sample Advice Logic

High GPU usage: Close heavy gaming or graphics-intensive applications

Charging during high load: Unplug the charger

Phone case present: Remove the phone case temporarily

Otherwise: Reduce screen brightness or allow the device to rest

Key Highlights

Combines regression and classification models

Handles class imbalance effectively

Safely manages feature mismatch during prediction

Models are saved and reused using .pkl files

Provides a real-time interactive user interface

Follows an industry-style machine learning workflow

Future Improvements

Time-series temperature prediction using LSTM models

Personalized models for different devices

On-device inference using TensorFlow Lite

Cloud deployment on platforms such as Render or Hugging Face Spaces

Integration with real sensor data