# save as generate_phone_temp_data.py
import numpy as np
import pandas as pd

np.random.seed(42)

N = 20000  # number of samples

# Ambient temp 15-40
ambient = np.random.normal(loc=30, scale=5, size=N)
ambient = np.clip(ambient, 10, 45)

cpu = np.clip(np.random.beta(2,5,size=N)*100 + np.random.normal(0,5,N), 0, 100)
gpu = np.clip(np.random.beta(2,6,size=N)*100 + np.random.normal(0,8,N), 0, 100)
ram = np.clip(np.random.beta(2,4,size=N)*100 + np.random.normal(0,6,N), 0, 100)
screen = np.clip(np.random.uniform(10,100,size=N) + np.random.normal(0,8,N), 0, 100)
battery = np.clip(100 - np.random.exponential(scale=30,size=N), 0, 100)
charging = np.random.binomial(1, 0.2, size=N)
case_on = np.random.binomial(1, 0.6, size=N)
fan = np.random.binomial(1, 0.05, size=N)
num_bg = np.random.poisson(2, size=N)
time_boot = np.clip(np.random.exponential(scale=120, size=N), 1, 1440)  # minutes

# network types and app load mapped to numeric intensity
network = np.random.choice(['wifi','4g','5g'], size=N, p=[0.5,0.35,0.15])
net_map = {'wifi':0.8, '4g':1.0, '5g':1.2}
net_multiplier = np.array([net_map[n] for n in network])

app = np.random.choice(['idle','social','video','game','camera'], size=N, p=[0.2,0.25,0.25,0.2,0.1])
app_map = {'idle':0.1, 'social':0.3, 'video':0.6, 'game':1.0, 'camera':0.7}
app_intensity = np.array([app_map[a] for a in app])

# Core physics-like temperature generation (toy model)
# baseline depends on ambient
base_temp = ambient + 5  # phone is warmer than ambient

# contributions
temp = (
    base_temp
    + 0.15 * cpu
    + 0.2 * gpu
    + 0.05 * ram
    + 0.02 * screen
    + 0.5 * app_intensity * 30  # apps effect
    + 3 * charging
    - 2 * fan
    + 2 * case_on
    + 0.01 * time_boot
    + 0.5 * num_bg
)

# network effect
temp = temp * (0.98 + 0.02 * net_multiplier)  # small network influence

# Add noise
temp += np.random.normal(0, 1.5, size=N)

# Clip to realistic range
temp = np.clip(temp, 20, 70)

# Threshold for overheating
threshold = 50.0
overheat = (temp >= threshold).astype(int)

df = pd.DataFrame({
    'ambient_temp': ambient,
    'cpu_usage': cpu,
    'gpu_usage': gpu,
    'ram_usage': ram,
    'screen_brightness': screen,
    'battery_level': battery,
    'charging': charging,
    'case_on': case_on,
    'fan': fan,
    'num_bg_apps': num_bg,
    'time_since_boot': time_boot,
    'network_type': network,
    'app_type': app,
    'app_intensity': app_intensity,
    'phone_temp': temp,
    'overheat': overheat
})

df.to_csv('phone_temp_dataset.csv', index=False)
print("Saved phone_temp_dataset.csv with", len(df), "rows")
