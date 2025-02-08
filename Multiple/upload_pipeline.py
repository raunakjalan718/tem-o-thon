import pymysql
import pandas as pd

# MySQL Database Connection
connection = pymysql.connect(
    host="localhost",   # Change if MySQL is on another server
    user="root",        # Change to your MySQL username
    password="hello123",  # Change to your MySQL password
    database="pipeline12"
)

cursor = connection.cursor()

# CSV File to Upload (Change filename for each pipeline)
csv_file = "pipeline12.csv"

# Read CSV
df = pd.read_csv(csv_file)

# Insert Data into MySQL
for _, row in df.iterrows():
    sql = """
    INSERT INTO pipeline_data (Timestamp, Flow_Inlet, Flow_Outlet)
    VALUES (%s, %s, %s)
    """
    values = (row["Timestamp"], row["Flow_Inlet (L/min)"], row["Flow_Outlet (L/min)"])
    
    cursor.execute(sql, values)

# Commit & Close
connection.commit()
cursor.close()
connection.close()

print(f"✅ {csv_file} uploaded successfully to MySQL!")
