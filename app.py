from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import json
import os
from datetime import datetime

app = Flask(__name__)

# ==========================================
# PATHS & LIGHTWEIGHT STORAGE ENGINE (JSON)
# ==========================================
DB_PATH = os.path.join("DataFiles", "database.json")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def init_db():
    """Initializes the JSON database file with template history data if missing."""
    if not os.path.exists(DB_PATH):
        initial_structure = {
            "screenings": [
                { "id": 1, "type": "parkinson", "name": "Parkinson Neurological Screening", "age": 68, "date": "Jun 14, 2026", "probability": 74, "risk": "high", "outcome": 1 },
                { "id": 2, "type": "heart", "name": "Heart Disease Screening", "age": 54, "date": "Jun 12, 2026", "probability": 18, "risk": "low", "outcome": 0 },
                { "id": 3, "type": "diabetes", "name": "Diabetes Disease Screening", "age": 33, "date": "Jun 12, 2026", "probability": 51, "risk": "moderate", "outcome": 1 },
                { "id": 4, "type": "heart", "name": "Heart Disease Screening", "age": 63, "date": "Jun 12, 2026", "probability": 82, "risk": "high", "outcome": 1 },
                { "id": 5, "type": "parkinson", "name": "Parkinson Neurological Screening", "age": 47, "date": "Jun 10, 2026", "probability": 12, "risk": "low", "outcome": 0 }
            ]
        }
        with open(DB_PATH, "w") as f:
            json.dump(initial_structure, f, indent=4)

def read_db():
    """Reads screening records from the persistent store file node."""
    init_db()
    with open(DB_PATH, "r") as f:
        return json.load(f)

def write_db(data):
    """Writes updated history matrix logs back to the disk database."""
    with open(DB_PATH, "w") as f:
        json.dump(data, f, indent=4)

def append_screening(screening_type, display_name, age, probability, risk_level, outcome):
    """Saves a prediction record to the top of the history list array."""
    db = read_db()
    new_id = max([s["id"] for s in db["screenings"]], default=0) + 1
    
    new_record = {
        "id": new_id,
        "type": screening_type,
        "name": display_name,
        "age": int(age),
        "date": datetime.now().strftime("%b %d, %Y"),
        "probability": int(round(probability * 100)),
        "risk": risk_level,
        "outcome": int(outcome)
    }
    db["screenings"].insert(0, new_record)  # Prepend so newest appears first
    write_db(db)


# ==========================================
# LOAD PIPELINE BINARIES
# ==========================================
heart_model = joblib.load("ML_Models/heart_disease_risk_pipeline.joblib")
diabetes_model = joblib.load("ML_Models/diabetes_pipeline.joblib")
parkinsons_model = joblib.load("ML_Models/parkinsons_pipeline.joblib")


# ==========================================
# PAGE ROUTING LINKS
# ==========================================
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/history")
def history():
    return render_template("history.html")

@app.route("/logout")
def logout():
    return render_template("index.html")

@app.route("/heart")
def heart():
    return render_template("heart.html")

@app.route("/diabetes")
def diabetes():
    return render_template("diabetes.html")

@app.route("/parkinsons")
def parkinsons():
    return render_template("parkinsons.html")


# ==========================================
# LIVE DATA INTERFACE API ENDPOINTS
# ==========================================
@app.route("/api/dashboard-metrics")
def get_dashboard_metrics():
    """Computes dynamic analytics summaries for your chart.js dashboard layouts."""
    db = read_db()
    screenings = db["screenings"]
    
    total = len(screenings)
    hearts = [s for s in screenings if s["type"] == "heart"]
    diabs = [s for s in screenings if s["type"] == "diabetes"]
    parks = [s for s in screenings if s["type"] == "parkinson"]
    
    high_risk = len([s for s in screenings if s["risk"] == "high"])
    
    # Calculate live positive diagnosis percent thresholds safely
    h_rate = round((len([s for s in hearts if s["outcome"] == 1]) / len(hearts)) * 100) if hearts else 0
    d_rate = round((len([s for s in diabs if s["outcome"] == 1]) / len(diabs)) * 100) if diabs else 0
    p_rate = round((len([s for s in parks if s["outcome"] == 1]) / len(parks)) * 100) if parks else 0

    # Extract distinct chronological dates for trend monitoring timelines (Limit to last 10 entries)
    raw_dates = sorted(list(set([s["date"] for s in screenings])))
    active_dates = raw_dates[-10:] if len(raw_dates) > 10 else raw_dates
    
    chart_labels = []
    heart_trend, diab_trend, park_trend = [], [], []
    
    for d in active_dates:
        try:
            dt_obj = datetime.strptime(d, "%b %d, %Y")
            chart_labels.append(dt_obj.strftime("%m-%d"))
        except:
            chart_labels.append(d)
            
        heart_trend.append(len([s for s in hearts if s["date"] == d]))
        diab_trend.append(len([s for s in diabs if s["date"] == d]))
        park_trend.append(len([s for s in parks if s["date"] == d]))

    return jsonify({
        "metrics": {
            "totalScreenings": total,
            "heartCount": len(hearts),
            "heartPositiveRate": h_rate,
            "diabetesCount": len(diabs),
            "diabetesPositiveRate": d_rate,
            "parkinsonCount": len(parks),
            "parkinsonPositiveRate": p_rate,
            "highRiskCases": high_risk
        },
        "trendChart": {
            "labels": chart_labels,
            "heartData": heart_trend,
            "diabetesData": diab_trend,
            "parkinsonData": park_trend
        },
        "riskDistribution": {
            "low": len([s for s in screenings if s["risk"] == "low"]),
            "moderate": len([s for s in screenings if s["risk"] == "moderate"]),
            "high": high_risk
        },
        "recentScreenings": screenings[:5]  # Serve latest 5 logs
    })

@app.route("/api/history-records")
def get_history_records():
    """Provides full history entries list array directly to table processors."""
    return jsonify(read_db())


# ==========================================
# HEART CLINICAL MODEL PREDICTOR ENDPOINT
# ==========================================
@app.route("/predict-heart", methods=["POST"])
def predict_heart():
    try:
        input_data = request.get_json()
        data = {
            "age": float(input_data["age"]), "sex": int(input_data["sex"]), "cp": int(input_data["cp"]),
            "trestbps": float(input_data["trestbps"]), "chol": float(input_data["chol"]), "fbs": int(input_data["fbs"]),
            "restecg": int(input_data["restecg"]), "thalach": float(input_data["thalach"]), "exang": int(input_data["exang"]),
            "oldpeak": float(input_data["oldpeak"]), "slope": int(input_data["slope"]), "ca": int(input_data["ca"]), "thal": int(input_data["thal"]),
        }

        df = pd.DataFrame([data])
        probability = float(heart_model.predict_proba(df)[0][1])
        target_outcome = 1 if probability >= 0.5 else 0
        risk_level = "low" if probability < 0.30 else ("moderate" if probability < 0.70 else "high")
        
        # Save prediction dynamically to JSON file database
        append_screening("heart", "Heart Disease Screening", data["age"], probability, risk_level, target_outcome)

        # Compute user relative variance deviations against standard reference points
        feature_means = {"age": 54.37, "sex": 0.68, "cp": 0.97, "trestbps": 131.62, "chol": 246.26, "fbs": 0.15, "restecg": 0.53, "thalach": 149.65, "exang": 0.33, "oldpeak": 1.04, "slope": 1.40, "ca": 0.73, "thal": 2.31}
        feature_stds = {"age": 9.08, "sex": 0.47, "cp": 1.03, "trestbps": 17.54, "chol": 51.83, "fbs": 0.36, "restecg": 0.53, "thalach": 22.91, "exang": 0.47, "oldpeak": 1.16, "slope": 0.62, "ca": 1.02, "thal": 0.61}
        labels = {"age": "Patient Age", "sex": "Sex", "cp": "Chest Pain Classification Type", "trestbps": "Resting Blood Pressure", "chol": "Serum Cholesterol Level", "fbs": "Fasting Blood Sugar threshold", "restecg": "Resting ECG Metrics", "thalach": "Maximum Heart Rate Achieved", "exang": "Exercise Induced Angina Findings", "oldpeak": "ST Depression (oldpeak)", "slope": "ST Segment Slope Profile", "ca": "Number of Major Colored Vessels", "thal": "Thalassemia Genetic Diagnostic Status"}
        
        contributions = []
        for key in data:
            z_score = abs((data[key] - feature_means[key]) / feature_stds[key])
            contributions.append({"key": key, "factor": labels[key], "score": z_score, "value": data[key]})
        
        contributions = sorted(contributions, key=lambda x: x["score"], reverse=True)

        top_risk_factors = []
        for idx, item in enumerate(contributions[:5]):
            impact = "high" if idx == 0 else ("medium" if idx <= 2 else "low")
            bar_width = "85%" if idx == 0 else ("60%" if idx <= 2 else "35%")
            v = item["value"]
            
            if item["key"] == "sex": val_str = "Male" if v == 1 else "Female"
            elif item["key"] == "cp": val_str = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][int(v)]
            elif item["key"] == "fbs": val_str = "Yes (>120 mg/dL)" if v == 1 else "No"
            elif item["key"] == "restecg": val_str = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"][int(v)]
            elif item["key"] == "exang": val_str = "Yes" if v == 1 else "No"
            elif item["key"] == "slope": val_str = ["Upsloping", "Flat", "Downsloping"][int(v)]
            elif item["key"] == "thal": val_str = ["Unknown", "Normal", "Fixed Defect", "Reversible Defect"][int(v)]
            elif item["key"] == "trestbps": val_str = f"{v} mm Hg"
            elif item["key"] == "chol": val_str = f"{v} mg/dL"
            elif item["key"] == "thalach": val_str = f"{v} bpm"
            elif item["key"] == "oldpeak": val_str = f"{float(v):.1f}"
            else: val_str = str(v)

            top_risk_factors.append({"factor": item["factor"], "impact": impact, "value": val_str, "barWidth": bar_width})

        recommendations = {
            "low": "Your results indicate a low risk. Maintain a heart-healthy lifestyle with regular exercise, a balanced diet, and routine check-ups.",
            "moderate": "Your results indicate a moderate risk profile. Schedule an appointment with your doctor to discuss lifestyle modifications, dietary adjustments, and possible routine screening testing.",
            "high": "Your results indicate a high risk of cardiovascular disease indicators. Please consult a cardiologist immediately for a comprehensive medical evaluation including ECG and structural stress metrics."
        }
        
        return jsonify({"probability": probability, "target": target_outcome, "riskLevel": risk_level, "topRiskFactors": top_risk_factors, "recommendation": recommendations[risk_level]})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ==========================================
# DIABETES CLINICAL PREDICTOR ENDPOINT
# ==========================================
@app.route("/predict-diabetes", methods=["POST"])
def predict_diabetes():
    try:
        input_data = request.get_json()
        data = {
            "Pregnancies": float(input_data["Pregnancies"]), "Glucose": float(input_data["Glucose"]), "BloodPressure": float(input_data["BloodPressure"]),
            "SkinThickness": float(input_data["SkinThickness"]), "Insulin": float(input_data["Insulin"]), "BMI": float(input_data["BMI"]),
            "DiabetesPedigreeFunction": float(input_data["DiabetesPedigreeFunction"]), "Age": float(input_data["Age"])
        }

        df = pd.DataFrame([data])
        probability = float(diabetes_model.predict_proba(df)[0][1])
        target_outcome = 1 if probability >= 0.5 else 0
        risk_level = "low" if probability < 0.30 else ("moderate" if probability < 0.70 else "high")
        
        # Save prediction dynamically to JSON file database
        append_screening("diabetes", "Diabetes Disease Screening", data["Age"], probability, risk_level, target_outcome)

        feature_means = {"Pregnancies": 3.84, "Glucose": 120.89, "BloodPressure": 69.10, "SkinThickness": 20.53, "Insulin": 79.79, "BMI": 31.99, "DiabetesPedigreeFunction": 0.471, "Age": 33.24}
        feature_stds = {"Pregnancies": 3.36, "Glucose": 31.97, "BloodPressure": 19.35, "SkinThickness": 15.95, "Insulin": 115.24, "BMI": 7.88, "DiabetesPedigreeFunction": 0.331, "Age": 11.76}
        labels = {"Pregnancies": "Pregnancy Count", "Glucose": "Plasma Glucose Concentration", "BloodPressure": "Diastolic Blood Pressure", "SkinThickness": "Triceps Skin Fold Thickness", "Insulin": "2-Hour Serum Insulin", "BMI": "Body Mass Index (BMI)", "DiabetesPedigreeFunction": "Diabetes Pedigree Scoring Index", "Age": "Patient Age"}
        
        contributions = []
        for key in data:
            z_score = abs((data[key] - feature_means[key]) / feature_stds[key])
            contributions.append({"key": key, "factor": labels[key], "score": z_score, "value": data[key]})
        
        contributions = sorted(contributions, key=lambda x: x["score"], reverse=True)
        
        top_risk_factors = []
        for idx, item in enumerate(contributions[:5]):
            impact = "high" if idx == 0 else ("medium" if idx <= 2 else "low")
            bar_width = "85%" if idx == 0 else ("60%" if idx <= 2 else "35%")
            
            val_str = str(item["value"])
            if item["key"] == "Glucose": val_str += " mg/dL"
            elif item["key"] == "BloodPressure": val_str += " mm Hg"
            elif item["key"] == "SkinThickness": val_str += " mm"
            elif item["key"] == "Insulin": val_str += " mu U/ml"
            elif item["key"] == "BMI": val_str += " kg/m²"
            elif item["key"] == "Age": val_str += " yrs"
            elif item["key"] == "Pregnancies": val_str += " times"

            top_risk_factors.append({"factor": item["factor"], "impact": impact, "value": val_str, "barWidth": bar_width})

        recommendations = {
            "low": "Your results show a stable baseline. Maintain balanced whole-food macro-nutritional patterns and general lifestyle health routines.",
            "moderate": "Your profile maps within a moderate metabolic pre-diabetic risk profile. Focus on routine monitoring alongside refined dietary glycemic index reduction parameters.",
            "high": "Your results show a significant indication of high diabetic risk trends. We highly advise a laboratory Fasting Plasma Glucose or HbA1c screening immediately."
        }

        return jsonify({"probability": probability, "Outcome": target_outcome, "riskLevel": risk_level, "topRiskFactors": top_risk_factors, "recommendation": recommendations[risk_level]})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ==========================================
# PARKINSON ACOUSTIC MODEL PREDICTOR ENDPOINT
# ==========================================
@app.route("/predict-parkinsons", methods=["POST"])
def predict_parkinsons():
    try:
        input_data = request.get_json()

        # Build data dictionary using original CSV schema mapping indices headers
        data = {
            "MDVP:Fo(Hz)": float(input_data["MDVP_Fo_Hz"]), "MDVP:Fhi(Hz)": float(input_data["MDVP_Fhi_Hz"]), "MDVP:Flo(Hz)": float(input_data["MDVP_Flo_Hz"]),
            "MDVP:Jitter(%)": float(input_data["MDVP_Jitter_Percent"]), "MDVP:Jitter(Abs)": float(input_data["MDVP_Jitter_Abs"]), "MDVP:RAP": float(input_data["MDVP_RAP"]),
            "MDVP:PPQ": float(input_data["MDVP_PPQ"]), "Jitter:DDP": float(input_data["Jitter_DDP"]), "MDVP:Shimmer": float(input_data["MDVP_Shimmer"]),
            "MDVP:Shimmer(dB)": float(input_data["MDVP_Shimmer_dB"]), "Shimmer:APQ3": float(input_data["Shimmer_APQ3"]), "Shimmer:APQ5": float(input_data["Shimmer_APQ5"]),
            "MDVP:APQ": float(input_data["MDVP_APQ"]), "Shimmer:DDA": float(input_data["Shimmer_DDA"]), "NHR": float(input_data["NHR"]), "HNR": float(input_data["HNR"]),
            "RPDE": float(input_data["RPDE"]), "DFA": float(input_data["DFA"]), "spread1": float(input_data["spread1"]), "spread2": float(input_data["spread2"]),
            "D2": float(input_data["D2"]), "PPE": float(input_data["PPE"])
        }

        df = pd.DataFrame([data])
        probability = float(parkinsons_model.predict_proba(df)[0][1])
        status_outcome = 1 if probability >= 0.5 else 0
        risk_level = "low" if probability < 0.35 else ("moderate" if probability < 0.65 else "high")
        
        # Save prediction dynamically to JSON file database (Approximate patient profile age context to 50)
        append_screening("parkinson", "Parkinson Neurological Screening", 50, probability, risk_level, status_outcome)

        feature_means = {"MDVP:Fo(Hz)": 154.228, "MDVP:Fhi(Hz)": 197.104, "MDVP:Flo(Hz)": 116.324, "MDVP:Jitter(%)": 0.0062, "MDVP:Jitter(Abs)": 0.000044, "MDVP:RAP": 0.0033, "MDVP:PPQ": 0.0034, "Jitter:DDP": 0.0099, "MDVP:Shimmer": 0.0297, "MDVP:Shimmer(dB)": 0.282, "Shimmer:APQ3": 0.0156, "Shimmer:APQ5": 0.0178, "MDVP:APQ": 0.0240, "Shimmer:DDA": 0.0469, "NHR": 0.0248, "HNR": 21.885, "RPDE": 0.498, "DFA": 0.718, "spread1": -5.684, "spread2": 0.226, "D2": 2.381, "PPE": 0.206}
        feature_stds = {"MDVP:Fo(Hz)": 41.390, "MDVP:Fhi(Hz)": 91.491, "MDVP:Flo(Hz)": 43.521, "MDVP:Jitter(%)": 0.0048, "MDVP:Jitter(Abs)": 0.000035, "MDVP:RAP": 0.0029, "MDVP:PPQ": 0.0027, "Jitter:DDP": 0.0089, "MDVP:Shimmer": 0.0188, "MDVP:Shimmer(dB)": 0.194, "Shimmer:APQ3": 0.0101, "Shimmer:APQ5": 0.0120, "MDVP:APQ": 0.0169, "Shimmer:DDA": 0.0304, "NHR": 0.0404, "HNR": 4.425, "RPDE": 0.103, "DFA": 0.055, "spread1": 1.090, "spread2": 0.083, "D2": 0.382, "PPE": 0.090}
        labels = {"MDVP:Fo(Hz)": "Average Fundamental Frequency (Fo)", "MDVP:Fhi(Hz)": "Max Fundamental Frequency (Fhi)", "MDVP:Flo(Hz)": "Min Fundamental Frequency (Flo)", "MDVP:Jitter(%)": "Vocal Jitter percentage", "MDVP:Jitter(Abs)": "Absolute Vocal Jitter", "MDVP:RAP": "Relative Amplitude Perturbation", "MDVP:PPQ": "Five-Point Period Perturbation", "Jitter:DDP": "Average Absolute Jitter Difference", "MDVP:Shimmer": "Local Shimmer Amplitude", "MDVP:Shimmer(dB)": "Local Shimmer Amplitude (dB)", "Shimmer:APQ3": "Three-Point Amplitude Perturbation", "Shimmer:APQ5": "Five-Point Amplitude Perturbation", "MDVP:APQ": "Eleven-Point Amplitude Perturbation", "Shimmer:DDA": "Average Absolute Shimmer Difference", "NHR": "Noise-to-Harmonic Ratio (NHR)", "HNR": "Harmonics-to-Noise Ratio (HNR)", "RPDE": "Recurrence Period Density Entropy", "DFA": "Detrended Fluctuation Analysis", "spread1": "Fundamental Frequency Variation 1", "spread2": "Fundamental Frequency Variation 2", "D2": "Correlation Dimension Measure", "PPE": "Pitch Period Entropy (PPE)"}

        contributions = []
        for key in data:
            z_score = abs((data[key] - feature_means[key]) / feature_stds[key])
            contributions.append({"key": key, "factor": labels[key], "score": z_score, "value": data[key]})

        contributions = sorted(contributions, key=lambda x: x["score"], reverse=True)

        top_risk_factors = []
        for idx, item in enumerate(contributions[:5]):
            impact = "high" if idx == 0 else ("medium" if idx <= 2 else "low")
            bar_width = "85%" if idx == 0 else ("60%" if idx <= 2 else "35%")
            v = item["value"]
            
            if item["key"] in ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)"]: val_str = f"{v:.2f} Hz"
            elif item["key"] in ["HNR", "MDVP:Shimmer(dB)"]: val_str = f"{v:.2f} dB"
            elif item["key"] == "MDVP:Jitter(%)": val_str = f"{(v * 100):.3f}%"
            else: val_str = f"{v:.4f}"

            top_risk_factors.append({"factor": item["factor"], "impact": impact, "value": val_str, "barWidth": bar_width})

        recommendations = {
            "low": "Acoustic frequency and micro-structural vectors fall cleanly within stable, expected parameters.",
            "moderate": "Measures register close to baseline boundary thresholds. We advise tracking phonatory profiles over a fixed time interval.",
            "high": "Vocal phonation markers show prominent acoustic period tremors. A clinical consultation with a primary neurologist is strongly recommended."
        }

        return jsonify({"probability": probability, "status": status_outcome, "riskLevel": risk_level, "topRiskFactors": top_risk_factors, "recommendation": recommendations[risk_level]})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ==========================================
# RE-START FLASK APPLICATION
# ==========================================
if __name__ == "__main__":
    app.run(debug=True)
    
# ==========================================
# FORCE CACHE CLEARING FOR NAVIGATION
# ==========================================
@app.after_request
def add_header(response):
    """
    Instructs the browser to completely bypass local cache memory.
    This guarantees fresh dynamic data loads during navbar tab swapping.
    """
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response    
    