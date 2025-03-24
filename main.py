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
    print(df.head())  # Print first 5 rows
else:
    print("No CSV files found in the dataset.")


