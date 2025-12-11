from sklearn.preprocessing import StandardScaler
from flask import Flask,request,render_template,redirect
import pandas as pd
import numpy
import os
import joblib
from pathlib import Path

# --- Configuration for Flask App ---
app = Flask(__name__)

# --- Load Model, Scaler, and Columns ---
# IMPORTANT: Verify the file names and paths match where you saved them 
# (e.g., RandomForest_Regressor_model.pkl, columns_rf.pkl)

try:
    # Attempt 1: Load from current working directory
    model = joblib.load(os.path.join(os.getcwd(), "RandomForest_Regressor_model.pkl"))
    scaler = joblib.load(os.path.join(os.getcwd(), "scaler.pkl"))
    columns = joblib.load(os.path.join(os.getcwd(), "columns_rf.pkl"))
except FileNotFoundError:
    # Attempt 2: Load from the specific 'projects/carbon emission prediction' subdirectory
    base_path = os.path.join(os.getcwd(), "projects", "carbon emission prediction")
    model = joblib.load(os.path.join(base_path, "randomforest_model.pkl"))
    scaler = joblib.load(os.path.join(base_path, "scaler.pkl"))
    columns = joblib.load(os.path.join(base_path, "columns.pkl"))


print("Model loaded successfully.")
print("Model type:", type(model))
print("Number of expected features:", len(columns))

# --- Flask Routes ---

@app.route("/")
def home():
    return redirect("/predictdata")

@app.route("/predictdata",methods=["GET","POST"])
def predict_datapoint():

    if request.method == "GET":
        # Assumes you have a home.html template ready
        return render_template("home.html")
        
    # --- STEP 1: Process and Normalize Inputs ---
    
    # Normalize categorical inputs to match training casing
    country_input = request.form.get("Country").strip().title()
    region_input = request.form.get("Region").strip().title()
    raw_date_input = request.form.get("Date") 
    
    data = {
        "Country": country_input,
        "Region": region_input,
        "Kilotons_of_Co2" : float(request.form.get("Kilotons_of_Co2")),
    }

    df = pd.DataFrame([data])
    
    # --- STEP 2: Feature Engineering (CRITICAL) ---
    
    # a. Date Feature Extraction (MUST MATCH TRAINING)
    try:
        # Assuming standard HTML date format YYYY-MM-DD
        df['Date_Temp'] = pd.to_datetime(raw_date_input)
    except ValueError:
        # Fallback for unexpected date format
        print(f"Warning: Could not parse date format '{raw_date_input}'. Defaulting date features.")
        df['Date_Temp'] = pd.to_datetime('2024-01-01')

    # Extract Year and Month features
    df['Year'] = df['Date_Temp'].dt.year            
    df['Month'] = df['Date_Temp'].dt.month          
    
    # b. Rename numeric feature to match training
    df.rename(columns={'Kilotons_of_Co2': 'Kilotons of Co2'}, inplace=True) 

    # c. Drop raw/temporary columns
    df.drop('Date_Temp', axis=1, inplace=True) 
    
    # --- STEP 3: One-Hot Encode ---
    df_clean = pd.get_dummies(df)

    # --- STEP 4: Align Columns with Training Data ---
    
    # 4a. Add missing columns (sets missing Country/Region features to 0)
    for col in columns:
        if col not in df_clean.columns:
            df_clean[col] = 0
            
    # 4b. Filter and reorder to the exact structure of the training columns. 
    df_clean = df_clean[columns]
    
    # --- STEP 5: Scale and Predict ---
    
    # Scaling on the cleaned and aligned DataFrame
    df_scaled = scaler.transform(df_clean)

    # Predict using the Random Forest Model
    predict = model.predict(df_scaled)[0]

    # Return the prediction formatted to 4 decimal places
    return f"Prediction : {predict:.4f} Metric Tons Per Capita"

if __name__ == "__main__":
    # Run the Flask app
    app.run(debug=True)