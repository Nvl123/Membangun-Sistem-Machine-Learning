import requests
import time
import random

# URL endpoint model ML (MLflow serving)
url = "http://localhost:5005/invocations"

# Headers untuk MLflow REST API
headers = {
    "Content-Type": "application/json"
}

# Kolom input yang sesuai dengan model
columns = ["CO", "NO2", "SO2", "O3", "PM2.5", "PM10"]

# Fungsi untuk membuat data acak (simulasi sensor)
def generate_random_input():
    return {
        "dataframe_split": {
            "columns": columns,
            "data": [[
                round(random.uniform(0.1, 2.0), 2),   # CO
                random.randint(10, 100),              # NO2
                random.randint(1, 20),                # SO2
                random.randint(10, 80),               # O3
                random.randint(10, 100),              # PM2.5
                random.randint(10, 150)               # PM10
            ]]
        }
    }

# Kirim request berulang
for i in range(20):  # Ubah jumlah iterasi sesuai kebutuhan
    payload = generate_random_input()
    try:
        response = requests.post(url, headers=headers, json=payload)
        print(f"[{i+1}] Status: {response.status_code}, Result: {response.json()}")
    except Exception as e:
        print(f"[{i+1}] Request failed: {e}")
    
    time.sleep(2)  # jeda 2 detik antar request
