from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# ==========================
# LOAD MODELS
# ==========================
heart_model = joblib.load("ML_Models/heart_disease_risk_pipeline.joblib")
diabetes_model = joblib.load("ML_Models/diabetes_pipeline.joblib")
parkinsons_model = joblib.load("ML_Models/parkinsons_pipeline.joblib")


# ==========================
# HOME
# ==========================
@app.route("/")
def home():
    return render_template("index.html")


# ==========================
# DASHBOARD
# ==========================
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


# ==========================
# HEART DISEASE PAGE
# ==========================
@app.route("/heart")
def heart():
    return render_template("heart.html")


@app.route("/predict-heart", methods=["POST"])
def predict_heart():

    data = {
        "age": float(request.form["age"]),
        "sex": int(request.form["sex"]),
        "cp": int(request.form["cp"]),
        "trestbps": float(request.form["trestbps"]),
        "chol": float(request.form["chol"]),
        "fbs": int(request.form["fbs"]),
        "restecg": int(request.form["restecg"]),
        "thalach": float(request.form["thalach"]),
        "exang": int(request.form["exang"]),
        "oldpeak": float(request.form["oldpeak"]),
        "slope": int(request.form["slope"]),
        "ca": int(request.form["ca"]),
        "thal": int(request.form["thal"]),
    }

    df = pd.DataFrame([data])

    probability = heart_model.predict_proba(df)[0][1]

    if probability < 0.30:
        level = "🟢 Low Risk"
    elif probability < 0.70:
        level = "🟡 Moderate Risk"
    else:
        level = "🔴 High Risk"

    return render_template(
        "heart.html",
        risk=round(probability * 100, 2),
        level=level
    )


# ==========================
# DIABETES PAGE
# ==========================
@app.route("/diabetes")
def diabetes():
    return render_template("diabetes.html")


@app.route("/predict-diabetes", methods=["POST"])
def predict_diabetes():

    data = {
        "Pregnancies": float(request.form["Pregnancies"]),
        "Glucose": float(request.form["Glucose"]),
        "BloodPressure": float(request.form["BloodPressure"]),
        "SkinThickness": float(request.form["SkinThickness"]),
        "Insulin": float(request.form["Insulin"]),
        "BMI": float(request.form["BMI"]),
        "DiabetesPedigreeFunction": float(request.form["DiabetesPedigreeFunction"]),
        "Age": float(request.form["Age"])
    }

    df = pd.DataFrame([data])

    probability = diabetes_model.predict_proba(df)[0][1]

    if probability < 0.30:
        level = "🟢 Low Risk"
    elif probability < 0.70:
        level = "🟡 Moderate Risk"
    else:
        level = "🔴 High Risk"

    return render_template(
        "diabetes.html",
        risk=round(probability * 100, 2),
        level=level
    )


# ==========================
# PARKINSON'S PAGE
# ==========================
@app.route("/parkinsons")
def parkinsons():
    return render_template("parkinsons.html")


@app.route("/predict-parkinsons", methods=["POST"])
def predict_parkinsons():

    data = {}

    for key in request.form:
        data[key] = float(request.form[key])

    df = pd.DataFrame([data])

    probability = parkinsons_model.predict_proba(df)[0][1]

    if probability < 0.30:
        level = "🟢 Low Risk"
    elif probability < 0.70:
        level = "🟡 Moderate Risk"
    else:
        level = "🔴 High Risk"

    return render_template(
        "parkinsons.html",
        risk=round(probability * 100, 2),
        level=level
    )


# ==========================
# RUN APP
# ==========================
if __name__ == "__main__":
    app.run(debug=True)