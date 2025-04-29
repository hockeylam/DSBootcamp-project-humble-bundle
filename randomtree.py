import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import os

#ML imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# Download the dataset
path = kagglehub.dataset_download("shrutibhargava94/india-air-quality-data")

print("Path to dataset files:", path)

# Get the first CSV file from the directory
csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]

if csv_files:
    file_path = os.path.join(path, csv_files[0])  # Get the first CSV file
    df = pd.read_csv(file_path, encoding="ISO-8859-1")  # Read CSV file into pandas DataFrame
    #print(df.head())  # Print first 5 rows
else:
    raise("error loading csv")

#delete problematic rows
df = df.dropna(subset=['date'])
df = df.dropna(subset=['state'])

#typecasting
df['stn_code'] = pd.to_numeric(df['stn_code'], errors='coerce')
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['month'] = df['date'].dt.month.astype(int, errors='raise')
df['year'] = df['date'].dt.year.astype(int, errors='raise')


#deleting redundant columns
delete_cols = ['stn_code', 'sampling_date', 'pm2_5', 'spm', 'date','agency']
for col in delete_cols:
    df = df.drop(col, axis=1)

print(df.head())

'''
selected_states = ['Puducherry', 'Goa', 'West Bengal']
df_filtered = df[df['year'] >= 1990]

# Compute SO2 averages for selected states
state_avg = df_filtered[df_filtered['state'].isin(selected_states)].groupby(['state', 'year'])['so2'].mean().reset_index()

# Compute SO2 average for all other states
other_states_avg = df_filtered[~df_filtered['state'].isin(selected_states)].groupby('year')['so2'].mean().reset_index()

# Plotting
plt.figure(figsize=(10, 6))

# Plot each selected state's SO2 levels
for state in selected_states:
    state_data = state_avg[state_avg['state'] == state]
    plt.plot(state_data['year'], state_data['so2'], label=state)

# Plot average SO2 for other states
plt.plot(other_states_avg['year'], other_states_avg['so2'], label='Other States Average', linestyle='--', color='black')

# Adding labels and title
plt.xlabel('Year')
plt.ylabel('Average SO2')
plt.title('Average SO2 Levels by State Over Time (Selected States vs Other States)')
plt.legend(loc='best', fontsize='small')
plt.grid(True)

# Display the plot
plt.show()
'''

'''

selected_states = ['West Bengal', 'Delhi', 'Rajasthan']

# Compute NO2 averages for selected states
state_avg = df[df['state'].isin(selected_states)].groupby(['state', 'year'])['no2'].mean().reset_index()

# Fill missing NO2 values using linear interpolation for each state
state_avg = state_avg.groupby('state').apply(lambda group: group.set_index('year').interpolate(method='linear').reset_index())

# Compute NO2 average for all other states
other_states_avg = df[~df['state'].isin(selected_states)].groupby('year')['no2'].mean().reset_index()

# Plotting
plt.figure(figsize=(10, 6))

# Plot each selected state's NO2 levels
for state in selected_states:
    state_data = state_avg[state_avg['state'] == state]
    plt.plot(state_data['year'], state_data['no2'], label=state)

# Plot average NO2 for other states
plt.plot(other_states_avg['year'], other_states_avg['no2'], label='Other States Average', linestyle='--', color='black')

# Adding labels and title
plt.xlabel('Year')
plt.ylabel('Average NO2')
plt.title('Average NO2 Levels by State Over Time (Selected States vs Other States)')
plt.legend(loc='best', fontsize='small')
plt.grid(True)

# Display the plot
plt.show()
'''
'''



selected_states = ['Haryana', 'Punjab', 'Delhi', 'Assam']
df_filtered = df[df['year'] >= 2004]

# Compute SO2 averages for selected states
state_avg = df_filtered[df_filtered['state'].isin(selected_states)].groupby(['state', 'year'])['rspm'].mean().reset_index()

# Compute SO2 average for all other states
other_states_avg = df_filtered[~df_filtered['state'].isin(selected_states)].groupby('year')['rspm'].mean().reset_index()

# Plotting
plt.figure(figsize=(10, 6))

# Plot each selected state's SO2 levels
for state in selected_states:
    state_data = state_avg[state_avg['state'] == state]
    plt.plot(state_data['year'], state_data['rspm'], label=state)

# Plot average SO2 for other states
plt.plot(other_states_avg['year'], other_states_avg['rspm'], label='Other States Average', linestyle='--', color='black')

# Adding labels and title
plt.xlabel('Year')
plt.ylabel('Average Rspm')
plt.title('Average Rspm Levels by State Over Time (Selected States vs Other States)')
plt.legend(loc='best', fontsize='small')
plt.grid(True)

# Display the plot
plt.show()'
'''

#machine learning portion:

#1. Prepare Data
# Drop rows where target pollutants are NaN
df = df.dropna(subset=['so2', 'no2', 'rspm'])

X = df.drop(['so2', 'no2', 'rspm'], axis=1) 
y_so2 = df['so2']
y_no2 = df['no2']
y_rspm = df['rspm']

#One-hot encodeing for areatype.
X = pd.get_dummies(X, columns=['type','state','location','month','year','location_monitoring_station'])

# 2. Train-test split
X_train, X_test, y_so2_train, y_so2_test = train_test_split(X, y_so2, test_size=0.2, random_state=42)
_, _, y_no2_train, y_no2_test = train_test_split(X, y_no2, test_size=0.2, random_state=42)
_, _, y_rspm_train, y_rspm_test = train_test_split(X, y_rspm, test_size=0.2, random_state=42)

# 3. Train a simple Linear Regression model
model_so2 = RandomForestRegressor(n_estimators=30, max_depth=10, random_state=42, n_jobs=-1)
model_no2 = RandomForestRegressor(n_estimators=30, max_depth=10, random_state=42, n_jobs=-1)
model_rspm = RandomForestRegressor(n_estimators=30, max_depth=10, random_state=42, n_jobs=-1)

model_so2.fit(X_train, y_so2_train)
model_no2.fit(X_train, y_no2_train)
model_rspm.fit(X_train, y_rspm_train)

# 4. Predictions
y_so2_pred = model_so2.predict(X_test)
y_no2_pred = model_no2.predict(X_test)
y_rspm_pred = model_rspm.predict(X_test)

# 5. Evaluate
print("SO2 Prediction Performance")
print("MSE:", mean_squared_error(y_so2_test, y_so2_pred))
print("R2:", r2_score(y_so2_test, y_so2_pred))

print("\nNO2 Prediction Performance")
print("MSE:", mean_squared_error(y_no2_test, y_no2_pred))
print("R2:", r2_score(y_no2_test, y_no2_pred))

print("\nRSPM Prediction Performance")
print("MSE:", mean_squared_error(y_rspm_test, y_rspm_pred))
print("R2:", r2_score(y_rspm_test, y_rspm_pred))

print("=======================================================")


#OVERFITTING CHECK
# Predictions on Training Set
y_so2_train_pred = model_so2.predict(X_train)
y_no2_train_pred = model_no2.predict(X_train)
y_rspm_train_pred = model_rspm.predict(X_train)

# Evaluate on Training Data
print("\n--- Overfitting Check (Training Data Performance) ---")

print("SO2 Training Performance")
print("Train MSE:", mean_squared_error(y_so2_train, y_so2_train_pred))
print("Train R2:", r2_score(y_so2_train, y_so2_train_pred))

print("\nNO2 Training Performance")
print("Train MSE:", mean_squared_error(y_no2_train, y_no2_train_pred))
print("Train R2:", r2_score(y_no2_train, y_no2_train_pred))

print("\nRSPM Training Performance")
print("Train MSE:", mean_squared_error(y_rspm_train, y_rspm_train_pred))
print("Train R2:", r2_score(y_rspm_train, y_rspm_train_pred))