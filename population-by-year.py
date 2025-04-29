import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import os

# ML imports
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# --- Download the initial datasets ---
path = kagglehub.dataset_download("shrutibhargava94/india-air-quality-data")
path2 = kagglehub.dataset_download("amritharj/population-of-india-19502022")
print("Path to air quality dataset files:", path)
print("Path to population dataset files:", path2)

# --- Load the air quality dataset ---
csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
if csv_files:
    file_path = os.path.join(path, csv_files[0])
    df = pd.read_csv(file_path, encoding="ISO-8859-1")
else:
    raise Exception("Error loading air quality csv")

# --- Load the population dataset ---
pop_csv_files = [f for f in os.listdir(path2) if f.endswith(".csv")]
if pop_csv_files:
    pop_file_path = os.path.join(path2, pop_csv_files[0])
    df_pop = pd.read_csv(pop_file_path)
else:
    raise Exception("Error loading population csv")

print("\nPopulation Data Sample:")
print(df_pop.head())

# --- Data Cleaning: Air Quality ---
df = df.dropna(subset=['date', 'state'])

df['stn_code'] = pd.to_numeric(df['stn_code'], errors='coerce')
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['month'] = df['date'].dt.month.astype(int, errors='raise')
df['year'] = df['date'].dt.year.astype(int, errors='raise')

# --- Data Cleaning: Population Dataset ---
for col in ['Population', 'Population Density']:
    if col in df_pop.columns:
        df_pop[col] = df_pop[col].replace({',': ''}, regex=True).astype(float)

if '% Increase in Population' in df_pop.columns:
    df_pop['% Increase in Population'] = df_pop['% Increase in Population'].str.replace('%', '', regex=False)
    df_pop['% Increase in Population'] = df_pop['% Increase in Population'].astype(float)

df_pop = df_pop.rename(columns={
    'Year': 'year',
    'Population': 'population',
    'Population Density': 'population_density',
    '% Increase in Population': 'percentage_increase_in_population'
})

# --- Merge datasets ---
df = df.merge(df_pop[['year', 'population', 'population_density', 'percentage_increase_in_population']],
              on=['year'],
              how='left')

# --- Delete redundant columns ---
delete_cols = ['stn_code', 'sampling_date', 'pm2_5', 'spm', 'date', 'agency']
for col in delete_cols:
    if col in df.columns:
        df = df.drop(col, axis=1)

print("\nExtended Air Quality Dataset Sample:")
with pd.option_context('display.max_columns', None):
    print(df.head())

# ============================================================
# --- Machine Learning: Simple Linear Regression (FAST) ---
# ============================================================

# 1. Prepare Data
df = df.dropna(subset=['so2', 'no2', 'rspm'])

X = df.drop(['so2', 'no2', 'rspm'], axis=1)
y_so2 = df['so2']
y_no2 = df['no2']
y_rspm = df['rspm']

# 2. Separate categorical and numeric features
categorical_features = ['type', 'state', 'location','location_monitoring_station']
numeric_features = ['month', 'year', 'population', 'population_density', 'percentage_increase_in_population']

X_categorical = pd.get_dummies(X[categorical_features])
X_numeric = X[numeric_features]

# Merge numeric and categorical
X_combined = pd.concat([X_numeric, X_categorical], axis=1)

# 3. Scale all features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)

# 4. Train-test split
X_train, X_test, y_so2_train, y_so2_test = train_test_split(X_scaled, y_so2, test_size=0.2, random_state=42)
_, _, y_no2_train, y_no2_test = train_test_split(X_scaled, y_no2, test_size=0.2, random_state=42)
_, _, y_rspm_train, y_rspm_test = train_test_split(X_scaled, y_rspm, test_size=0.2, random_state=42)

# 5. Define and Train Linear Regression models
model_so2 = LinearRegression()
model_so2.fit(X_train, y_so2_train)
y_so2_pred = model_so2.predict(X_test)
print("SO2 Prediction: MSE =", mean_squared_error(y_so2_test, y_so2_pred), "| R2 =", r2_score(y_so2_test, y_so2_pred))

model_no2 = LinearRegression()
model_no2.fit(X_train, y_no2_train)
y_no2_pred = model_no2.predict(X_test)
print("NO2 Prediction: MSE =", mean_squared_error(y_no2_test, y_no2_pred), "| R2 =", r2_score(y_no2_test, y_no2_pred))

model_rspm = LinearRegression()
model_rspm.fit(X_train, y_rspm_train)
y_rspm_pred = model_rspm.predict(X_test)
print("RSPM Prediction: MSE =", mean_squared_error(y_rspm_test, y_rspm_pred), "| R2 =", r2_score(y_rspm_test, y_rspm_pred))

# 6. Overfitting Check (Training Data)
print("\n--- Training Set Performance ---")

y_so2_train_pred = model_so2.predict(X_train)
print("SO2 Training: MSE =", mean_squared_error(y_so2_train, y_so2_train_pred), "| R2 =", r2_score(y_so2_train, y_so2_train_pred))

y_no2_train_pred = model_no2.predict(X_train)
print("NO2 Training: MSE =", mean_squared_error(y_no2_train, y_no2_train_pred), "| R2 =", r2_score(y_no2_train, y_no2_train_pred))

y_rspm_train_pred = model_rspm.predict(X_train)
print("RSPM Training: MSE =", mean_squared_error(y_rspm_train, y_rspm_train_pred), "| R2 =", r2_score(y_rspm_train, y_rspm_train_pred))
