import pandas as pd
from sqlalchemy import create_engine

# MySQL Database Credentials
DB_USER = "root"
DB_PASSWORD = "hello123"
DB_HOST = "localhost"

# Create engine for `input_mean`
engine = create_engine(f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/input_mean")

# Load mean values
query = "SELECT * FROM mean_values2"
df = pd.read_sql(query, engine)

# Display table
print(df)

# Save as CSV (optional)
df.to_csv("mean_values.csv", index=False)
print("Mean values saved as mean_values.csv")
