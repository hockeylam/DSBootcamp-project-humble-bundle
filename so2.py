# Imports
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import kagglehub

# Step 1: Load air quality dataset
path = kagglehub.dataset_download("shrutibhargava94/india-air-quality-data")
csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
if csv_files:
    file_path = os.path.join(path, csv_files[0])
    df = pd.read_csv(file_path, encoding="ISO-8859-1", low_memory=False)
else:
    raise Exception("CSV not found")

# Step 2: Normalize column names
df.columns = [col.lower().strip().replace('.', '_') for col in df.columns]

# Step 3: Preprocessing
df = df.dropna(subset=['date', 'state', 'location', 'so2', 'no2', 'rspm'])
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year
df = df.drop(columns=['stn_code', 'sampling_date', 'spm', 'agency', 'date'], errors='ignore')

# Step 4: Fill numeric values just in case (minimize loss)
for col in ['so2', 'no2', 'rspm']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col].fillna(df[col].median(), inplace=True)

# Step 5: Map state to rainfall district
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

# Step 6: Load rainfall dataset
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

# Map month names to integers
month_map = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
             'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}
rain_long['month'] = rain_long['month_str'].map(month_map)
rain_long.rename(columns={'YEAR': 'year', 'SUBDIVISION': 'district'}, inplace=True)

# Merge rainfall
df = pd.merge(df, rain_long[['district', 'year', 'month', 'rainfall']], on=['district', 'year', 'month'], how='left')
df['rainfall'] = df['rainfall'].fillna(df['rainfall'].median())

# Step 7: Define features and target (include rainfall!)
X = df[['year', 'month', 'no2', 'rspm', 'state', 'location', 'rainfall']]
y = df['so2']

# Drop rows with any missing values
df_model = pd.concat([X, y], axis=1).dropna()
X = df_model.drop(columns=['so2'])
y = df_model['so2']

# Step 8: Preprocessing pipeline for categorical features
categorical_cols = ['state', 'location']
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
], remainder='passthrough')

# Step 9: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 10: High-capacity model for best R² under 4 mins
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(
        n_estimators=100,      # increase trees
        max_depth=40,   # very fine splits
        n_jobs=-1,             # full CPU usage
        random_state=42
    ))
])
model.fit(X_train, y_train)

# Step 11: Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("Model Evaluation Metrics (High Accuracy, Rainfall Included):")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# Step 12: Plot results
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, edgecolor='k', label='Predictions')
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel("Actual SO₂")
plt.ylabel("Predicted SO₂")
plt.title("Random Forest: Actual vs Predicted SO₂ (with Rainfall)")
plt.grid(True)
plt.legend()
plt.show()
