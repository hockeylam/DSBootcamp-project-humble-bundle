import kagglehub
import pandas as pd
import os
import matplotlib.pyplot as plt

#ML imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# --- Step 1: Load air quality dataset ---
print("Loading air quality data...")
path = kagglehub.dataset_download("shrutibhargava94/india-air-quality-data")
csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
file_path = os.path.join(path, csv_files[0])
df = pd.read_csv(file_path, encoding="ISO-8859-1")
print("✔ Air quality data loaded.")

# --- Step 2: Clean air quality data ---
df = df.dropna(subset=['date', 'state'])
df['stn_code'] = pd.to_numeric(df['stn_code'], errors='coerce')
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year
df = df.drop(columns=['stn_code', 'sampling_date', 'pm2_5', 'spm', 'date', 'agency'], errors='ignore')

# --- Step 3: Map state to district ---
state_to_district = {
    'Assam': 'ASSAM & MEGHALAYA', 'Meghalaya': 'ASSAM & MEGHALAYA', 'Arunachal Pradesh': 'ASSAM & MEGHALAYA',
    'Nagaland': 'ASSAM & MEGHALAYA', 'Manipur': 'ASSAM & MEGHALAYA', 'Tripura': 'ASSAM & MEGHALAYA', 'Mizoram': 'ASSAM & MEGHALAYA',
    'Jammu & Kashmir': 'NAGA MANI MIZO TRIPURA', 'Himachal Pradesh': 'NAGA MANI MIZO TRIPURA',
    'Uttarakhand': 'NAGA MANI MIZO TRIPURA', 'Uttaranchal': 'NAGA MANI MIZO TRIPURA',
    'Sikkim': 'NAGA MANI MIZO TRIPURA', 'Delhi': 'NAGA MANI MIZO TRIPURA', 'Chandigarh': 'NAGA MANI MIZO TRIPURA',
    'Haryana': 'NAGA MANI MIZO TRIPURA', 'Punjab': 'NAGA MANI MIZO TRIPURA',
    'West Bengal': 'SUB HIMALAYAN WEST BENGAL & SIKKIM', 'Jharkhand': 'SUB HIMALAYAN WEST BENGAL & SIKKIM',
    'Bihar': 'SUB HIMALAYAN WEST BENGAL & SIKKIM', 'Uttar Pradesh': 'SUB HIMALAYAN WEST BENGAL & SIKKIM',
    'Odisha': 'GANGETIC WEST BENGAL', 'Chhattisgarh': 'GANGETIC WEST BENGAL',
    'Madhya Pradesh': 'GANGETIC WEST BENGAL', 'Rajasthan': 'GANGETIC WEST BENGAL',
    'Gujarat': 'ORISSA', 'Maharashtra': 'ORISSA', 'Andhra Pradesh': 'ORISSA', 'Telangana': 'ORISSA',
    'Tamil Nadu': 'ORISSA', 'Karnataka': 'ORISSA', 'Kerala': 'ORISSA', 'Goa': 'ORISSA',
    'Dadra & Nagar Haveli': 'ORISSA', 'Daman & Diu': 'ORISSA', 'Puducherry': 'ORISSA'
}
df['district'] = df['state'].map(state_to_district)

# --- Step 4: Load rainfall data and reshape ---
print("Loading rainfall data...")
rain_path = kagglehub.dataset_download("thedevastator/annual-subdivision-wise-rainfall-in-india-1901-2")
rain_file = [f for f in os.listdir(rain_path) if f.endswith(".csv")][0]
rain_df = pd.read_csv(os.path.join(rain_path, rain_file))

# Reshape wide to long
rain_long = pd.melt(
    rain_df,
    id_vars=['SUBDIVISION', 'YEAR'],
    value_vars=['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'],
    var_name='month_str',
    value_name='rainfall'
)
month_map = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
             'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}
rain_long['month'] = rain_long['month_str'].map(month_map)
rain_long.rename(columns={'YEAR': 'year', 'SUBDIVISION': 'district'}, inplace=True)

# --- Step 5: Merge rainfall with air quality data ---
df = pd.merge(df, rain_long[['district', 'year', 'month', 'rainfall']], on=['district', 'year', 'month'], how='left')

print(df.head())

# --- Step 6: Drop NaN targets and prepare data ---
df = df.dropna(subset=['so2', 'no2', 'rspm'])
X = df.drop(['so2', 'no2', 'rspm'], axis=1).copy()
y_so2 = df['so2']
y_no2 = df['no2']
y_rspm = df['rspm']

# Make sure rainfall is retained
if 'rainfall' not in X.columns:
    X['rainfall'] = df['rainfall']

# --- Step 7: One-hot encode and train ---
X = pd.get_dummies(X, columns=['type', 'state', 'location', 'month', 'year', 'location_monitoring_station', 'district','rainfall'])

# Split
X_train, X_test, y_so2_train, y_so2_test = train_test_split(X, y_so2, test_size=0.2, random_state=42)
_, _, y_no2_train, y_no2_test = train_test_split(X, y_no2, test_size=0.2, random_state=42)
_, _, y_rspm_train, y_rspm_test = train_test_split(X, y_rspm, test_size=0.2, random_state=42)

# Train models
model_so2 = LinearRegression().fit(X_train, y_so2_train)
model_no2 = LinearRegression().fit(X_train, y_no2_train)
model_rspm = LinearRegression().fit(X_train, y_rspm_train)

# Evaluate
print("\n=== Test Set Performance ===")
print("SO2 → MSE:", mean_squared_error(y_so2_test, model_so2.predict(X_test)), " | R²:", r2_score(y_so2_test, model_so2.predict(X_test)))
print("NO2 → MSE:", mean_squared_error(y_no2_test, model_no2.predict(X_test)), " | R²:", r2_score(y_no2_test, model_no2.predict(X_test)))
print("RSPM → MSE:", mean_squared_error(y_rspm_test, model_rspm.predict(X_test)), " | R²:", r2_score(y_rspm_test, model_rspm.predict(X_test)))

# Overfitting check
print("\n=== Training Set Performance ===")
print("SO2 → MSE:", mean_squared_error(y_so2_train, model_so2.predict(X_train)), " | R²:", r2_score(y_so2_train, model_so2.predict(X_train)))
print("NO2 → MSE:", mean_squared_error(y_no2_train, model_no2.predict(X_train)), " | R²:", r2_score(y_no2_train, model_no2.predict(X_train)))
print("RSPM → MSE:", mean_squared_error(y_rspm_train, model_rspm.predict(X_train)), " | R²:", r2_score(y_rspm_train, model_rspm.predict(X_train)))

print("\n✅ Done: Rainfall is now included as a feature.")
