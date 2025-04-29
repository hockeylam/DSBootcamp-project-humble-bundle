import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import os

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# --- Download the datasets ---
path = kagglehub.dataset_download("shrutibhargava94/india-air-quality-data")
path2 = kagglehub.dataset_download("rdatta871/population-of-india")
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

# Typecasting
df['stn_code'] = pd.to_numeric(df['stn_code'], errors='coerce')
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['month'] = df['date'].dt.month.astype(int, errors='raise')
df['year'] = df['date'].dt.year.astype(int, errors='raise')

# --- Classify months into seasons ---
def classify_season(month):
    if month in [12, 1, 2]:
        return 'winter'
    elif month in [3, 4, 5]:
        return 'summer'
    elif month in [6, 7, 8, 9]:
        return 'monsoon'
    elif month in [10, 11]:
        return 'post_monsoon'
    else:
        return 'unknown'

df['season'] = df['month'].apply(classify_season)

# --- Data Cleaning: Population Dataset ---
# Remove commas and cast to float
for col in ['Population', 'Urban population', 'Rural population']:
    if col in df_pop.columns:
        df_pop[col] = df_pop[col].replace({',': ''}, regex=True).astype(float)

# Rename columns
df_pop = df_pop.rename(columns={
    'State/UT': 'state',
    'Population[50]': 'population',
    'Urban[51]': 'urban_population',
    'Rural[51]': 'rural_population'
})

# Lowercase and strip whitespace for matching
df['state'] = df['state'].str.strip().str.lower()
df_pop['state'] = df_pop['state'].str.strip().str.lower()

# --- Merge datasets on state (not year anymore) ---
df = df.merge(df_pop[['state', 'population', 'urban_population', 'rural_population']],
              on=['state'],
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
# --- Machine Learning Part ---
# ============================================================

# --- Fix: Drop all rows with NaN (after merging population and adding season) ---
df = df.dropna()
print(f"Dataset shape after dropping NaNs: {df.shape}")

# Prepare Data
X = df.drop(['so2', 'no2', 'rspm'], axis=1)
y_so2 = df['so2']
y_no2 = df['no2']
y_rspm = df['rspm']

# One-hot encode only categorical features (not numeric features like population!)
X = pd.get_dummies(X, columns=['type', 'state', 'location', 'month', 'year', 'season', 'location_monitoring_station'])

# Train-test split
X_train, X_test, y_so2_train, y_so2_test = train_test_split(X, y_so2, test_size=0.2, random_state=42)
_, _, y_no2_train, y_no2_test = train_test_split(X, y_no2, test_size=0.2, random_state=42)
_, _, y_rspm_train, y_rspm_test = train_test_split(X, y_rspm, test_size=0.2, random_state=42)

# Train Linear Regression models
model_so2 = LinearRegression()
model_no2 = LinearRegression()
model_rspm = LinearRegression()

model_so2.fit(X_train, y_so2_train)
model_no2.fit(X_train, y_no2_train)
model_rspm.fit(X_train, y_rspm_train)

# Predictions
y_so2_pred = model_so2.predict(X_test)
y_no2_pred = model_no2.predict(X_test)
y_rspm_pred = model_rspm.predict(X_test)

# Evaluation
print("\n--- Test Set Performance ---")
print("SO2 Prediction: MSE =", mean_squared_error(y_so2_test, y_so2_pred), "| R2 =", r2_score(y_so2_test, y_so2_pred))
print("NO2 Prediction: MSE =", mean_squared_error(y_no2_test, y_no2_pred), "| R2 =", r2_score(y_no2_test, y_no2_pred))
print("RSPM Prediction: MSE =", mean_squared_error(y_rspm_test, y_rspm_pred), "| R2 =", r2_score(y_rspm_test, y_rspm_pred))

# Overfitting Check (Training Data)
y_so2_train_pred = model_so2.predict(X_train)
y_no2_train_pred = model_no2.predict(X_train)
y_rspm_train_pred = model_rspm.predict(X_train)

print("\n--- Training Set Performance (Overfitting Check) ---")
print("SO2 Training: MSE =", mean_squared_error(y_so2_train, y_so2_train_pred), "| R2 =", r2_score(y_so2_train, y_so2_train_pred))
print("NO2 Training: MSE =", mean_squared_error(y_no2_train, y_no2_train_pred), "| R2 =", r2_score(y_no2_train, y_no2_train_pred))
print("RSPM Training: MSE =", mean_squared_error(y_rspm_train, y_rspm_train_pred), "| R2 =", r2_score(y_rspm_train, y_rspm_train_pred))
