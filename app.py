# app.py
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify

# Initialize the Flask app
app = Flask(__name__)

# --- Load Models and Metadata ONCE at startup ---
try:
    model_weather = joblib.load('model_weather.joblib')
    model_env = joblib.load('model_env.joblib')
    model_weather_cols = joblib.load('model_weather_columns.joblib')
    model_env_cols = joblib.load('model_env_columns.joblib')
    unique_summaries = joblib.load('unique_summaries.joblib')
except FileNotFoundError:
    print("Error: Model files not found. Please run 'train_model.py' first.")
    exit()

# --- Route 1: The Home Page ---
# This serves your index.html file
@app.route('/')
def home():
    # Pass the list of summaries to the HTML template to build the dropdown
    return render_template('index.html', summaries=unique_summaries)

# --- Route 2: API for Model 1 (Predict Weather) ---
@app.route('/predict_weather', methods=['POST'])
def predict_weather():
    try:
        # Get data from the web form (sent as JSON)
        data = request.json
        
        # Create a DataFrame in the exact format the model expects
        user_input = [[
            float(data['temp']),
            float(data['humidity']),
            float(data['pressure']),
            float(data['wind_speed'])
        ]]
        user_input_df = pd.DataFrame(user_input, columns=model_weather_cols)
        
        # Make prediction
        prediction = model_weather.predict(user_input_df)[0]
        
        # Determine the action string (same logic as your notebook)
        if "Rain" in prediction:
            action = "Carry an umbrella ☔"
        elif "Clear" in prediction or "Sunny" in prediction:
            action = "Wear sunglasses 😎"
        elif "Cloud" in prediction:
            action = "Stay updated with weather news ☁️"
        else:
            action = "Check local forecast for details 🌦️"
            
        # Send the result back as JSON
        return jsonify({'prediction': prediction, 'action': action})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# --- Route 3: API for Model 2 (Predict Details) ---
@app.route('/predict_details', methods=['POST'])
def predict_details():
    try:
        # Get data from the web form
        data = request.json
        temp = float(data['temp'])
        summary = data['summary']
        
        # --- Recreate the One-Hot Encoding ---
        # Create a DataFrame with all possible columns, initialized to 0
        encoded_input = pd.DataFrame(0, index=[0], columns=model_env_cols)
        
        # Set the temperature
        encoded_input['Temperature (C)'] = temp
        
        # Set the selected weather type column to 1
        weather_col = f"Summary_{summary}"
        if weather_col in encoded_input.columns:
            encoded_input[weather_col] = 1
        else:
            # Handle case where summary might be unknown (though dropdown prevents this)
            print(f"Warning: Summary '{summary}' not in model columns.")

        # Ensure column order is exactly the same as during training
        encoded_input = encoded_input[model_env_cols]
        
        # Make prediction
        pred = model_env.predict(encoded_input)[0]
        
        # Send the result back as JSON
        return jsonify({
            'humidity': f"{pred[0]:.2f}",
            'pressure': f"{pred[1]:.2f}",
            'wind_speed': f"{pred[2]:.2f}"
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True)