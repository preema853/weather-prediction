# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import joblib # Import joblib to save models
from sklearn.metrics import accuracy_score

print("Starting model training...")

# --- 1. Load and Prepare Data (from your Cell 1) ---
try:
    data = pd.read_csv(r"C:\Users\AIMIT\Downloads\weather\weatherHistory.csv")
except FileNotFoundError:
    print("Error: The file 'weatherHistory.csv' was not found.")
    print("Please make sure the file path is correct.")
    exit()

# --- Data for Model 1: Predict Weather Type ---
X1 = data[['Temperature (C)', 'Humidity', 'Pressure (millibars)', 'Wind Speed (km/h)']]
y1 = data['Summary']
X1 = X1.fillna(X1.mean())
y1 = y1.fillna('Unknown')

# --- Data for Model 2: Predict Environment Details ---
data2 = data[['Temperature (C)', 'Summary', 'Humidity', 'Pressure (millibars)', 'Wind Speed (km/h)']].copy()
data2.dropna(inplace=True)
unique_summaries = sorted(data2['Summary'].unique())
data2_encoded = pd.get_dummies(data2, columns=['Summary'])
X2 = data2_encoded.drop(columns=['Humidity', 'Pressure (millibars)', 'Wind Speed (km/h)'])
y2 = data2_encoded[['Humidity', 'Pressure (millibars)', 'Wind Speed (km/h)']]

print("Data loaded and prepared.")

# --- 2. Model Training (from your Cell 2) ---
# --- Train Model 1: Predict Weather Type ---
model_weather = DecisionTreeClassifier(random_state=42)
model_weather.fit(X1, y1)
print("Model 1 (Weather Type Predictor) training complete.")

# --- Train Model 2: Predict Environment Details ---
model_env = DecisionTreeRegressor(random_state=42)
model_env.fit(X2, y2)
print("Model 2 (Environment Details Predictor) training complete.")

# --- 3. Save Models and Metadata ---
joblib.dump(model_weather, 'model_weather.joblib')
joblib.dump(model_env, 'model_env.joblib')

# Save the column lists and summaries, which are critical for the API
joblib.dump(list(X1.columns), 'model_weather_columns.joblib')
joblib.dump(list(X2.columns), 'model_env_columns.joblib')
joblib.dump(unique_summaries, 'unique_summaries.joblib')

print("All models and metadata have been saved successfully!")
