import pandas as pd

# Load the dataset
df = pd.read_csv("leakage_detection_dataset.csv")  # Ensure file is in the same directory

# Display first few rows
print(df.head())


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Features (input variables) and target variable (output)
X = df[['Flow_A (L/min)', 'Flow_C (L/min)', 'Temperature (°C)', 'Time_of_Day (hr)']]
y = df['Leakage']

# Split into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Initialize Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)  # Train the model

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Print classification report
print(classification_report(y_test, y_pred))

import joblib

# Save the trained model
joblib.dump(model, "leakage_detection_model.pkl")

# Load the saved model
loaded_model = joblib.load("leakage_detection_model.pkl")
print("Model Loaded Successfully!")


# Example new input data
new_data = scaler.transform([[8.0, 7.1, 30, 14]])

# Predict leakage
prediction = loaded_model.predict(new_data)
print("Leakage Detected" if prediction[0] == 1 else "No Leakage Detected")


import matplotlib.pyplot as plt
import seaborn as sns

# Calculate Flow Difference
df['Flow_Diff'] = df['Flow_A (L/min)'] - df['Flow_C (L/min)']

# Create a box plot
plt.figure(figsize=(8, 6))
sns.boxplot(x=df['Leakage'], y=df['Flow_Diff'])
plt.xlabel("Leakage (0 = No, 1 = Yes)")
plt.ylabel("Flow Difference (A - C)")
plt.title("Water Flow Difference vs. Leakage")
plt.show()
