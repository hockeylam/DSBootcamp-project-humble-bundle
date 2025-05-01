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

# Step 1: Load dataset
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

# Step 4: Define features and target
X = df[['year', 'month', 'so2', 'no2', 'state', 'location']]
y = df['rspm']

# Drop rows with any missing values in X or y
df_model = pd.concat([X, y], axis=1).dropna()
X = df_model.drop(columns=['rspm'])
y = df_model['rspm']

# Step 5: Preprocessing for categorical columns
categorical_cols = ['state', 'location']
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
], remainder='passthrough')

# Step 6: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
df_model = df_model.sample(n=5000, random_state=42)  # Optional subsample

# Step 7: Build and train model pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=30, max_depth=40, n_jobs=-1, random_state=42)
)
])
model.fit(X_train, y_train)

# Step 8: Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("Model Evaluation Metrics:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# Step 9: Plot results
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, edgecolor='k', label='Predictions')
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel("Actual RSPM")
plt.ylabel("Predicted RSPM")
plt.title("Random Forest: Actual vs Predicted RSPM")
plt.grid(True)
plt.legend()
plt.show()
