import pandas as pd
import numpy as np
import mysql.connector
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

# MySQL Database Credentials
DB_USER = "root"
DB_PASSWORD = "hello123"
DB_HOST = "localhost"

# Create SQLAlchemy engine for `input_mean` database
engine_input_mean = create_engine(f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/input_mean")

# List to store accuracy results
accuracy_results = {}

# Dictionary to store hourly mean values
hourly_means_dict = {}

# Train and process data for each pipeline
for pipeline_id in range(1, 13):
    database_name = f"pipeline{pipeline_id}"
    
    # Create SQLAlchemy engine for the current pipeline database
    engine_pipeline = create_engine(f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{database_name}")
    
    # Load data
    query = "SELECT timestamp, Flow_Inlet, Flow_Outlet FROM pipeline_data"
    df = pd.read_sql(query, engine_pipeline)
    
    # Convert timestamp to datetime format
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour  # Extract hour

    # Compute Flow Difference
    df["flow_difference"] = df["Flow_Inlet"] - df["Flow_Outlet"]

    # Initialize list to store means for this pipeline
    hourly_means = []

    # Train model for each hour (0 to 23)
    for hour in range(24):
        hourly_data = df[df["hour"] == hour]["flow_difference"]
        
        if not hourly_data.empty:
            mean_value = np.mean(hourly_data)  # Mean of flow difference for this hour
            std_dev = np.std(hourly_data)  # Standard deviation
            accuracy = round(100 - (std_dev / mean_value * 100), 2) if mean_value != 0 else 100
            
            # Store accuracy for this hour and pipeline
            if hour not in accuracy_results:
                accuracy_results[hour] = {}
            accuracy_results[hour][f"pipeline{pipeline_id}"] = accuracy
            
        else:
            mean_value = None
        
        hourly_means.append(mean_value)

    # Store means in dictionary
    hourly_means_dict[f"pipeline{pipeline_id}"] = hourly_means

    # Plot graph for this pipeline
    plt.figure(figsize=(10, 5))
    plt.plot(range(24), hourly_means, marker="o", linestyle="-", label=f"Pipeline {pipeline_id}")
    plt.xlabel("Hour")
    plt.ylabel("Mean Flow Difference")
    plt.title(f"Flow Difference per Hour - Pipeline {pipeline_id}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"pipeline{pipeline_id}_graph.png")  # Save the plot
    plt.close()

# Convert the dictionary into a DataFrame
df_means = pd.DataFrame.from_dict(hourly_means_dict, orient="index").transpose()
df_means.reset_index(inplace=True)
df_means.rename(columns={"index": "hour"}, inplace=True)

# Store data in `mean_values2` table in `input_mean` database
df_means.to_sql("mean_values2", engine_input_mean, if_exists="replace", index=False)

# Print accuracy results
for hour in range(24):
    print(f"Hour {hour}: {accuracy_results[hour]}")

print("Model training complete, mean values stored, and graphs saved!")
