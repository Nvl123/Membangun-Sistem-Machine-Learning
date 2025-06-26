import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import joblib
import os

# Set tracking URI ke server MLflow
mlflow.set_tracking_uri("http://localhost:5000")

# Muat dataset
df = pd.read_csv("air_quality_cleaned.csv")

# Pisahkan fitur dan target
X = df[['CO', 'NO2', 'SO2', 'O3', 'PM2.5', 'PM10']]  # Fitur
y = df['AQI']  # Target

# Split dataset menjadi data training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Dataset info untuk logging
dataset_info = {
    'total_samples': len(df),
    'features': list(X.columns),
    'target_column': 'AQI',
    'train_size': len(X_train),
    'test_size': len(X_test),
    'test_ratio': 0.2,
    'random_state': 42,
    'missing_values': df.isnull().sum().sum(),
    'feature_count': X.shape[1]
}

# Direktori untuk menyimpan artifact
artifact_dir = "/app/model_artifacts"
# Pastikan direktori ada
os.makedirs(artifact_dir, exist_ok=True)

# Mendefinisikan parameter untuk hyperparameter tuning
models_to_test = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(random_state=42),
    'Lasso': Lasso(random_state=42, max_iter=2000)
}

# Parameter tuning untuk Ridge dan Lasso
alpha_range = np.logspace(-4, 2, 10)  # Alpha dari 0.0001 hingga 100

# Variabel untuk menyimpan model terbaik
best_r2 = -np.inf
best_params = {}
best_model = None
best_model_name = ""

print("Memulai hyperparameter tuning...")
print("="*50)

# Loop untuk setiap jenis model
for model_name, base_model in models_to_test.items():
    print(f"\nTesting {model_name}...")
    
    if model_name == 'LinearRegression':
        # Linear Regression tidak memiliki hyperparameter untuk di-tune
        run_name = f"{model_name}"
        
        with mlflow.start_run(run_name=run_name):
            # MANUAL LOGGING - TIDAK menggunakan autolog
            
            # Train model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Prediksi dan evaluasi
            y_pred = model.predict(X_test)
            
            # Hitung metrik
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            # MANUAL LOG METRICS
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("rmse", rmse)
            
            # MANUAL LOG PARAMETERS
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("fit_intercept", model.fit_intercept)
            mlflow.log_param("copy_X", model.copy_X)
            mlflow.log_param("n_jobs", model.n_jobs)
            mlflow.log_param("positive", model.positive)
            
            # MANUAL LOG DATASET INFO
            for key, value in dataset_info.items():
                mlflow.log_param(f"dataset_{key}", value)
            
            # MANUAL LOG MODEL ATTRIBUTES (setelah training)
            mlflow.log_param("n_features_in", model.n_features_in_)
            mlflow.log_param("feature_names", str(X.columns.tolist()))
            
            # Log coefficients dan intercept
            for i, coef in enumerate(model.coef_):
                mlflow.log_param(f"coef_{X.columns[i]}", coef)
            mlflow.log_param("intercept", model.intercept_)
            
            # MANUAL SAVE dan LOG ARTIFACTS
            # Simpan model sebagai file .pkl
            model_pkl_path = os.path.join(artifact_dir, f"model_{model_name}.pkl")
            joblib.dump(model, model_pkl_path)
            
            # Simpan dataset info sebagai artifact
            dataset_info_path = os.path.join(artifact_dir, f"dataset_info_{model_name}.txt")
            with open(dataset_info_path, 'w') as f:
                f.write("DATASET INFORMATION\n")
                f.write("="*50 + "\n")
                for key, value in dataset_info.items():
                    f.write(f"{key}: {value}\n")
                f.write("\nDATASET STATISTICS\n")
                f.write("="*50 + "\n")
                f.write(f"Target (AQI) Statistics:\n")
                f.write(f"Mean: {y.mean():.4f}\n")
                f.write(f"Std: {y.std():.4f}\n")
                f.write(f"Min: {y.min():.4f}\n")
                f.write(f"Max: {y.max():.4f}\n")
                f.write(f"\nFeature Statistics:\n")
                f.write(df[X.columns].describe().to_string())
            
            # Simpan dataset sample sebagai CSV
            dataset_sample_path = os.path.join(artifact_dir, f"dataset_sample_{model_name}.csv")
            df.head(100).to_csv(dataset_sample_path, index=False)
            
            # Simpan artifact dari file yang sudah ada di dalam container
            artifact_files = ["conda.yaml", "python_env.yaml", "requirements.txt"]
            for artifact_file in artifact_files:
                artifact_path = os.path.join(artifact_dir, artifact_file)
                if os.path.exists(artifact_path):
                    mlflow.log_artifact(artifact_path)
            
            # Log semua artifacts
            mlflow.log_artifact(model_pkl_path)
            mlflow.log_artifact(dataset_info_path)
            mlflow.log_artifact(dataset_sample_path)
            
            # MANUAL LOG MODEL dalam format MLflow
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=None
            )
            
            print(f"  R2: {r2:.4f}, RMSE: {rmse:.4f}")
            print(f"  Manual logging selesai untuk {run_name}")
            
            # Cek apakah ini model terbaik
            if r2 > best_r2:
                best_r2 = r2
                best_params = {"model": model_name}
                best_model = model
                best_model_name = model_name
    
    else:
        # Untuk Ridge dan Lasso, lakukan tuning alpha
        for alpha in alpha_range:
            run_name = f"{model_name}_alpha_{alpha:.6f}"
            
            with mlflow.start_run(run_name=run_name):
                # MANUAL LOGGING - TIDAK menggunakan autolog
                
                # Inisialisasi model dengan alpha spesifik
                if model_name == 'Ridge':
                    model = Ridge(alpha=alpha, random_state=42)
                else:  # Lasso
                    model = Lasso(alpha=alpha, random_state=42, max_iter=2000)
                
                # Train model
                model.fit(X_train, y_train)
                
                # Prediksi dan evaluasi
                y_pred = model.predict(X_test)
                
                # Hitung metrik
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mse)
                
                # MANUAL LOG METRICS
                mlflow.log_metric("mse", mse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2", r2)
                mlflow.log_metric("rmse", rmse)
                
                # MANUAL LOG PARAMETERS
                mlflow.log_param("model_type", model_name)
                mlflow.log_param("alpha", alpha)
                mlflow.log_param("fit_intercept", model.fit_intercept)
                mlflow.log_param("copy_X", model.copy_X)
                mlflow.log_param("random_state", model.random_state)
                
                # MANUAL LOG DATASET INFO
                for key, value in dataset_info.items():
                    mlflow.log_param(f"dataset_{key}", value)
                
                if model_name == 'Ridge':
                    mlflow.log_param("solver", model.solver)
                    mlflow.log_param("tol", model.tol)
                else:  # Lasso
                    mlflow.log_param("max_iter", model.max_iter)
                    mlflow.log_param("tol", model.tol)
                    mlflow.log_param("warm_start", model.warm_start)
                    mlflow.log_param("positive", model.positive)
                    mlflow.log_param("selection", model.selection)
                
                # MANUAL LOG MODEL ATTRIBUTES (setelah training)
                mlflow.log_param("n_features_in", model.n_features_in_)
                mlflow.log_param("feature_names", str(X.columns.tolist()))
                mlflow.log_param("n_iter", getattr(model, 'n_iter_', 'N/A'))
                
                # Log coefficients dan intercept
                for i, coef in enumerate(model.coef_):
                    mlflow.log_param(f"coef_{X.columns[i]}", coef)
                mlflow.log_param("intercept", model.intercept_)
                
                # MANUAL SAVE dan LOG ARTIFACTS
                # Simpan model sebagai file .pkl
                model_pkl_path = os.path.join(artifact_dir, f"model_{model_name}_alpha_{alpha:.6f}.pkl")
                joblib.dump(model, model_pkl_path)
                
                # Simpan dataset info sebagai artifact
                dataset_info_path = os.path.join(artifact_dir, f"dataset_info_{model_name}_alpha_{alpha:.6f}.txt")
                with open(dataset_info_path, 'w') as f:
                    f.write("DATASET INFORMATION\n")
                    f.write("="*50 + "\n")
                    for key, value in dataset_info.items():
                        f.write(f"{key}: {value}\n")
                    f.write("\nDATASET STATISTICS\n")
                    f.write("="*50 + "\n")
                    f.write(f"Target (AQI) Statistics:\n")
                    f.write(f"Mean: {y.mean():.4f}\n")
                    f.write(f"Std: {y.std():.4f}\n")
                    f.write(f"Min: {y.min():.4f}\n")
                    f.write(f"Max: {y.max():.4f}\n")
                    f.write(f"\nFeature Statistics:\n")
                    f.write(df[X.columns].describe().to_string())
                
                # Simpan dataset sample sebagai CSV
                dataset_sample_path = os.path.join(artifact_dir, f"dataset_sample_{model_name}_alpha_{alpha:.6f}.csv")
                df.head(100).to_csv(dataset_sample_path, index=False)
                
                # Simpan artifact dari file yang sudah ada di dalam container
                artifact_files = ["conda.yaml", "python_env.yaml", "requirements.txt"]
                for artifact_file in artifact_files:
                    artifact_path = os.path.join(artifact_dir, artifact_file)
                    if os.path.exists(artifact_path):
                        mlflow.log_artifact(artifact_path)
                
                # Log semua artifacts
                mlflow.log_artifact(model_pkl_path)
                mlflow.log_artifact(dataset_info_path)
                mlflow.log_artifact(dataset_sample_path)
                
                # MANUAL LOG MODEL dalam format MLflow
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    registered_model_name=None
                )
                
                print(f"  Alpha: {alpha:.6f}, R2: {r2:.4f}, RMSE: {rmse:.4f}")
                print(f"  Manual logging selesai untuk {run_name}")
                
                # Cek apakah ini model terbaik
                if r2 > best_r2:
                    best_r2 = r2
                    best_params = {"model": model_name, "alpha": alpha}
                    best_model = model
                    best_model_name = f"{model_name}_alpha_{alpha:.6f}"

print("\n" + "="*50)
print("HYPERPARAMETER TUNING SELESAI")
print("="*50)
print(f"Model terbaik: {best_params}")
print(f"R2 terbaik: {best_r2:.4f}")

# Evaluasi ulang model terbaik untuk mendapatkan semua metrik
y_pred_best = best_model.predict(X_test)

mse_best = mean_squared_error(y_test, y_pred_best)
mae_best = mean_absolute_error(y_test, y_pred_best)
rmse_best = np.sqrt(mse_best)

print(f"\nMetrik model terbaik:")
print(f"MSE: {mse_best:.4f}")
print(f"MAE: {mae_best:.4f}")
print(f"R2: {best_r2:.4f}")
print(f"RMSE: {rmse_best:.4f}")
print(f"Parameter: {best_params}")