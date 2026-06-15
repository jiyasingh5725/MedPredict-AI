import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib

# =========================
# PATHS
# =========================
DATA_PATH = Path("DataFiles/heart_disease_data.csv")
MODEL_PATH = Path("ML_Models/heart_disease_risk_pipeline.joblib")
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATA_PATH)

target_col = "target"
X = df.drop(columns=[target_col])
y = df[target_col]

# =========================
# FEATURE GROUPS
# =========================
numeric_feats = ["age", "trestbps", "chol", "thalach", "oldpeak"]

categorical_feats = [
    "sex", "cp", "fbs", "restecg",
    "exang", "slope", "ca", "thal"
]

# =========================
# PREPROCESSING
# =========================
numeric_transformer = StandardScaler()

categorical_transformer = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_feats),
        ("cat", categorical_transformer, categorical_feats),
    ]
)

# =========================
# MODEL
# =========================
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

# =========================
# SPLIT DATA
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# =========================
# TRAIN
# =========================
clf.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

print("\n✅ Accuracy:", round(accuracy_score(y_test, y_pred), 4))

try:
    print("📊 ROC AUC:", round(roc_auc_score(y_test, y_proba), 4))
except:
    pass

print("\n📄 Classification Report:\n")
print(classification_report(y_test, y_pred))

# =========================
# SAVE MODEL
# =========================
joblib.dump(clf, MODEL_PATH)
print(f"\n💾 Model saved at: {MODEL_PATH}")

# =========================
# RISK FUNCTIONS
# =========================
def get_risk_level(prob):
    if prob < 0.30:
        return "🟢 Low Risk"
    elif prob < 0.70:
        return "🟡 Moderate Risk"
    else:
        return "🔴 High Risk"

def predict_risk(sample_df):
    """
    Input: DataFrame (same columns as training data)
    Output: list of dicts with % + risk level
    """
    prob = clf.predict_proba(sample_df)[:, 1]

    results = []
    for p in prob:
        results.append({
            "risk_percent": round(p * 100, 2),
            "risk_level": get_risk_level(p)
        })

    return results


# =========================
# TEST EXAMPLE
# =========================
if __name__ == "__main__":
    sample = X_test.iloc[[0]]

    print("\n🧪 Sample Input:")
    print(sample)

    print("\n❤️ Prediction:")
    print(predict_risk(sample))