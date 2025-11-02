# 🌦️ Weather Prediction System

This project is a **Weather Prediction Web Application** built using **Python (Flask)**.  
It predicts future weather conditions (such as temperature, humidity, and weather type)  
based on historical environmental data.

---

## 🚀 Features
- Predicts weather using trained machine learning models (`.joblib` files).  
- User-friendly web interface (HTML templates).  
- Displays predictions instantly after entering input data.  
- Uses real-world weather datasets for model training.  

---

## 🧠 Tech Stack
- **Frontend:** HTML, CSS (Flask templates)  
- **Backend:** Python, Flask  
- **Machine Learning:** Scikit-learn  
- **Models Saved As:** `.joblib` files

  ---

## 🧩 Project Structure
├── app.py # Flask app
├── train_model.py # Script to train and save models
├── templates/ # HTML templates
├── model_env.joblib # Saved environment model
├── model_weather.joblib # Saved weather model
├── weatherHistory.xlsx # Weather dataset (optional)
└── .gitignore


---

## ⚙️ How It Works
1. The model is trained using past weather data.  
2. The trained model files (`.joblib`) are loaded by `app.py`.  
3. When the user enters input (like temperature, humidity, etc.),  
   the model predicts the weather condition.  
4. The prediction is displayed on the web page instantly.

---
