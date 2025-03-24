import kagglehub
import pandas as pd
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








