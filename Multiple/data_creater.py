import pandas as pd
import numpy as np

date_rng = pd.date_range(start="2024-01-01", periods=1008, freq="10T")

flow_in = np.random.uniform(20, 25, len(date_rng))  
flow_out = flow_in - np.random.uniform(0, 5, len(date_rng))  

df = pd.DataFrame({
    "Timestamp": date_rng,
    "Flow_Inlet (L/min)": flow_in,
    "Flow_Outlet (L/min)": flow_out
})

df.to_csv("pipeline12.csv", index=False)
print("✅ pipeline12.csv has been created successfully!")
