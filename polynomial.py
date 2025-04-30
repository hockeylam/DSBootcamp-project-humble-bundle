import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA

print("Step 1: Downloading dataset...")
path = kagglehub.dataset_download("shrutibhargava94/india-air-quality-data")
print("✔ Dataset downloaded at:", path)

# Load the air quality dataset
print("Step 2: Loading CSV file...")
csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
if not csv_files:
    raise Exception("❌ No CSV file found in the dataset path.")
file_path = os.path.join(path, csv_files[0])
df = pd.read_csv(file_path, encoding="ISO-8859-1", low_memory=False, dtype={0: str})
print("✔ DataFrame loaded successfully.")

# Step 3: Cleaning data
print("Step 3: Cleaning and preparing data...")
df = df.dropna(subset=['date', 'state', 'location', 'type'])

# Convert and clean date
df['stn_code'] = pd.to_numeric(df['stn_code'], errors='coerce')
df['location_monitoring_station'] = pd.to_numeric(df['location_monitoring_station'], errors='coerce')
df['location_monitoring_station'] = df['location_monitoring_station'].fillna(0).astype(int)  # Fix: fill NaNs with 0
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['month'] = df['date'].dt.month.astype(int, errors='raise')
df['year'] = df['date'].dt.year.astype(int, errors='raise')

# Create combined location-station ID
df['location_station'] = df['location'].astype(str) + '-' + df['location_monitoring_station'].astype(str)

# Drop unnecessary columns
delete_cols = ['stn_code', 'sampling_date', 'pm2_5', 'spm', 'date', 'agency']
df = df.drop(columns=delete_cols, errors='ignore')
print("✔ Cleaned and created 'location_station' column.")

# Step 4: Drop rows with missing target values
print("Step 4: Dropping rows with missing target values...")
df = df.dropna(subset=['so2', 'no2', 'rspm'])

# Step 5: Prepare features and targets
print("Step 5: Preparing features and targets...")
X_numeric = df[['month', 'year']]
y_so2 = df['so2']
y_no2 = df['no2']
y_rspm = df['rspm']

# Polynomial transformation on numeric features
print("Step 6: Applying Polynomial Features to numeric features...")
X_numeric = X_numeric.dropna()
poly = PolynomialFeatures(degree=2, include_bias=False)
X_numeric_poly = poly.fit_transform(X_numeric)
print("✔ Polynomial transformation complete.")
print("Shape of numeric poly:", X_numeric_poly.shape)

# Align categorical and target data with cleaned numeric indices
valid_idx = X_numeric.index
df = df.loc[valid_idx]
y_so2 = y_so2.loc[valid_idx]
y_no2 = y_no2.loc[valid_idx]
y_rspm = y_rspm.loc[valid_idx]

# One-hot encode selected features (state, type, location, location_station)
X_categorical = pd.get_dummies(df[['state', 'type', 'location', 'location_station']], drop_first=True)
print("✔ One-hot encoding complete.")
print("Categorical feature shape:", X_categorical.shape)

# Step 6.1: Combine features
X_combined = np.hstack([X_numeric_poly, X_categorical.values])
print("Combined feature shape before scaling:", X_combined.shape)

# Step 6.2: Scale features
print("Step 6.2: Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)
print("✔ Scaling complete.")

# Step 6.3: Apply PCA
print("Step 6.3: Applying PCA...")
pca = PCA(n_components=0.95)  # keep 95% of variance
X_final = pca.fit_transform(X_scaled)
print("✔ PCA transformation complete.")
print("Final shape after PCA:", X_final.shape)

# Step 7: Train-test split
print("Step 7: Splitting data into train and test sets...")
X_train, X_test, y_so2_train, y_so2_test = train_test_split(X_final, y_so2, test_size=0.2, random_state=42)
_, _, y_no2_train, y_no2_test = train_test_split(X_final, y_no2, test_size=0.2, random_state=42)
_, _, y_rspm_train, y_rspm_test = train_test_split(X_final, y_rspm, test_size=0.2, random_state=42)
print("✔ Data split complete.")

# Step 8: Train and evaluate Linear Regression models
print("Step 8: Training Linear Regression models...")

model_so2 = LinearRegression()
model_so2.fit(X_train, y_so2_train)
y_so2_pred = model_so2.predict(X_test)

model_no2 = LinearRegression()
model_no2.fit(X_train, y_no2_train)
y_no2_pred = model_no2.predict(X_test)

model_rspm = LinearRegression()
model_rspm.fit(X_train, y_rspm_train)
y_rspm_pred = model_rspm.predict(X_test)

# Results
print("\n=== Test Set Performance ===")
print("SO2 → MSE:", mean_squared_error(y_so2_test, y_so2_pred), " | R²:", r2_score(y_so2_test, y_so2_pred))
print("NO2 → MSE:", mean_squared_error(y_no2_test, y_no2_pred), " | R²:", r2_score(y_no2_test, y_no2_pred))
print("RSPM → MSE:", mean_squared_error(y_rspm_test, y_rspm_pred), " | R²:", r2_score(y_rspm_test, y_rspm_pred))

# Overfitting check
print("\n=== Training Set Performance (Overfitting Check) ===")
y_so2_train_pred = model_so2.predict(X_train)
print("SO2 → Train MSE:", mean_squared_error(y_so2_train, y_so2_train_pred), " | Train R²:", r2_score(y_so2_train, y_so2_train_pred))

y_no2_train_pred = model_no2.predict(X_train)
print("NO2 → Train MSE:", mean_squared_error(y_no2_train, y_no2_train_pred), " | Train R²:", r2_score(y_no2_train, y_no2_train_pred))

y_rspm_train_pred = model_rspm.predict(X_train)
print("RSPM → Train MSE:", mean_squared_error(y_rspm_train, y_rspm_train_pred), " | Train R²:", r2_score(y_rspm_train, y_rspm_train_pred))

print("\n✅ All done.")
