import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import os

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
delete_cols = ['agency', 'stn_code', 'location', 'sampling_date', 'location_monitoring_station', 'pm2_5', 'spm', 'date']
for col in delete_cols:
    df = df.drop(col, axis=1)

print(df.head())

'''

SO2


selected_states = ['Puducherry', 'Goa', 'West Bengal', 'Bihar', 'Gujarat']
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

NO2

selected_states = ['Puducherry', 'West Bengal', 'Delhi', 'Rajasthan']

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

RSPM


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
plt.show()
'''













