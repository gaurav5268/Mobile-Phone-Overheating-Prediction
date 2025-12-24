import streamlit as st
import pandas as pd
import numpy as np
import joblib

reg_model = joblib.load("temperature_regressor.pkl")
clf_model = joblib.load("overheat_classifier.pkl")
feature_columns = joblib.load("model_features.pkl")

st.set_page_config(page_title="Mobile Overheating Predictor", layout="centered")

st.title("📱 Mobile Phone Overheating Prediction System")
st.write("Predict phone temperature, overheating risk and get safety advice.")

st.header("🔧 Phone Usage Inputs")

ambient_temp = st.slider("Ambient Temperature (°C)", 20, 45, 30)
cpu_usage = st.slider("CPU Usage (%)", 0, 100, 50)
gpu_usage = st.slider("GPU Usage (%)", 0, 100, 40)
ram_usage = st.slider("RAM Usage (%)", 0, 100, 60)
screen_brightness = st.slider("Screen Brightness (%)", 0, 100, 70)
battery_level = st.slider("Battery Level (%)", 0, 100, 50)
num_bg_apps = st.slider("Background Apps", 0, 15, 5)
time_since_boot = st.slider("Time Since Boot (minutes)", 1, 500, 120)

charging = st.selectbox("Charging?", ["No", "Yes"])
case_on = st.selectbox("Phone Case On?", ["No", "Yes"])
fan = st.selectbox("Cooling Fan?", ["No", "Yes"])

app_type = st.selectbox("App Type", ["idle", "social", "video", "game"])
network_type = st.selectbox("Network Type", ["wifi", "4g", "5g"])

if st.button("🔍 Predict"):

    charging = 1 if charging == "Yes" else 0
    case_on = 1 if case_on == "Yes" else 0
    fan = 1 if fan == "Yes" else 0

    raw_input = {
        'ambient_temp': ambient_temp,
        'cpu_usage': cpu_usage,
        'gpu_usage': gpu_usage,
        'ram_usage': ram_usage,
        'screen_brightness': screen_brightness,
        'battery_level': battery_level,
        'charging': charging,
        'case_on': case_on,
        'fan': fan,
        'num_bg_apps': num_bg_apps,
        'time_since_boot': time_since_boot,
        'app_intensity': 1.0,
        'app_type': app_type,
        'network_type': network_type
    }

    raw_df = pd.DataFrame([raw_input])

    raw_df = pd.get_dummies(raw_df)

    raw_df['cpu_gpu_mean'] = (cpu_usage + gpu_usage) / 2
    raw_df['high_load'] = int(cpu_usage > 70)
    raw_df['hot_ambient'] = int(ambient_temp > 35)
    raw_df['boot_log'] = np.log1p(time_since_boot)
    raw_df['cpu_case_interaction'] = cpu_usage * case_on

    raw_df = raw_df.reindex(columns=feature_columns, fill_value=0)

    pred_temp = reg_model.predict(raw_df)[0]
    overheat_prob = clf_model.predict_proba(raw_df)[0][1]

    if pred_temp >= 50 or overheat_prob >= 0.4:
        alert = "🔥 OVERHEATING RISK"
    else:
        alert = "✅ SAFE"

    if gpu_usage > 80:
        advice = "Close heavy gaming or graphics apps."
    elif cpu_usage > 75:
        advice = "Close background apps."
    elif charging == 1:
        advice = "Unplug charger to reduce heat."
    elif case_on == 1:
        advice = "Remove phone case temporarily."
    else:
        advice = "Reduce screen brightness or rest the phone."

    st.header("📊 Prediction Result")

    st.metric("Predicted Temperature (°C)", f"{pred_temp:.2f}")
    st.metric("Overheat Probability", f"{overheat_prob:.2f}")
    st.subheader(alert)
    st.info(f"💡 Advice: {advice}")