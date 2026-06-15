import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib

# =========================
# PATHS
# =========================
DATA_PATH = Path("DataFiles/parkinsons.csv")
MODEL_PATH = Path("ML_Models/parkinsons_pipeline.joblib")
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATA_PATH)

# Remove patient name if present
if "name" in df.columns:
    df = df.drop(columns=["name"])

target_col = "status"

X = df.drop(columns=[target_col])
y = df[target_col]

# =========================
# PREPROCESSING
# =========================
numeric_feats = X.columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_feats)
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

clf = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# =========================
# SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# =========================
# TRAIN
# =========================
clf.fit(X_train,y_train)

# =========================
# EVALUATION
# =========================
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:,1]

print("Accuracy:", accuracy_score(y_test,y_pred))
print("ROC AUC:", roc_auc_score(y_test,y_prob))
print(classification_report(y_test,y_pred))

# =========================
# SAVE MODEL
# =========================
joblib.dump(clf, MODEL_PATH)
print("Model saved:", MODEL_PATH)

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
    probs = clf.predict_proba(sample_df)[:,1]
    return [
        {
            "risk_percent": round(p*100,2),
            "risk_level": get_risk_level(p)
        }
        for p in probs
    ]

if __name__=="__main__":
    sample = X_test.iloc[[0]]
    print(predict_risk(sample))