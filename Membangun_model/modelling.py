import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Gunakan MLflow Tracking lokal
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # atau "http://localhost:5000"

# Aktifkan autolog MLflow untuk scikit-learn
mlflow.sklearn.autolog()

# Muat dataset
df = pd.read_csv("air_quality_cleaned.csv")

# Pisahkan fitur dan target
X = df[['CO', 'NO2', 'SO2', 'O3', 'PM2.5', 'PM10']]
y = df['AQI']

# Bagi data menjadi train dan test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Mulai run MLflow
with mlflow.start_run(run_name="LinearRegression_with_autolog"):
    # Melatih model Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluasi otomatis dilakukan oleh autolog
    print("Training selesai. Semua artefak, parameter, dan metrik dicatat otomatis.")
