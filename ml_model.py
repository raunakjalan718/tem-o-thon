import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib  # For saving and loading the model

# 1️⃣ Generate Synthetic Data (Simulating Water Flow Sensor Readings)
np.random.seed(42)  # Ensures reproducibility
data_size = 500  # Number of data points

Flow_A = np.random.uniform(5, 10, data_size)  # Flow rate at sensor A (L/min)
Flow_C = Flow_A - np.random.uniform(0, 2, data_size)  # Flow rate at sensor C (L/min)
Temperature = np.random.uniform(20, 40, data_size)  # Temperature in °C
Time_of_Day = np.random.randint(0, 24, data_size)  # Time in hours (0-23)

# Leakage Label: 1 (Leakage) if Flow difference > 1.5 L/min, else 0 (No Leakage)
Leakage = np.where((Flow_A - Flow_C) > 1.5, 1, 0)

# 2️⃣ Create DataFrame
df = pd.DataFrame({
    "Flow_A (L/min)": Flow_A,
    "Flow_C (L/min)": Flow_C,
    "Temperature (°C)": Temperature,
    "Time_of_Day (hr)": Time_of_Day,
    "Leakage": Leakage
})

# 3️⃣ Split Data into Training and Testing Sets
X = df.drop(columns=["Leakage"])  # Features
y = df["Leakage"]  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4️⃣ Normalize Features for Better Accuracy
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use same scaler for test set

# 5️⃣ Train the Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# 6️⃣ Evaluate Model Performance
y_pred = rf_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.2f}\n")

print("🔎 Classification Report:")
print(classification_report(y_test, y_pred))

# 7️⃣ Save the Trained Model and Scaler
joblib.dump(rf_model, "leakage_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("✅ Model and Scaler Saved!")

# 8️⃣ Load the Model for Prediction
rf_model = joblib.load("leakage_model.pkl")
scaler = joblib.load("scaler.pkl")
print("✅ Model Loaded Successfully!\n")

# 9️⃣ Predict Leakage for a New Sensor Reading
# New data input (Simulating real-time sensor data)
X_new = np.array([7.5, 6.0, 25, 10])  # Flow_A, Flow_C, Temperature, Time

# ✅ FIX: Convert X_new to a DataFrame with Correct Feature Names
feature_names = ["Flow_A (L/min)", "Flow_C (L/min)", "Temperature (°C)", "Time_of_Day (hr)"]
X_new_df = pd.DataFrame([X_new], columns=feature_names)  # Ensuring same feature names

# Standardize using the trained scaler
X_new_scaled = scaler.transform(X_new_df)

# Make prediction
prediction = rf_model.predict(X_new_scaled)

# Display Result
if prediction[0] == 1:
    print("⚠️ Leakage Detected!")
else:
    print("✅ No Leakage Detected!")
