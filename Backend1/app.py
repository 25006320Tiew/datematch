"""
DateMatch Prediction API
Backend for the Dating App Match Outcome Predictor

Pipeline (mirrors the notebook exactly):
  1. Categorical features  → LabelEncoder (per-column)
  2. All 15 features       → StandardScaler
  3. GradientBoostingClassifier → predict + predict_proba
  4. Encoded int           → LabelEncoder (target) inverse_transform
"""

import os, sys, warnings
import numpy as np
import joblib
from flask import Flask, jsonify, request
from flask_cors import CORS

warnings.filterwarnings("ignore", category=UserWarning)

# ── sklearn version guard ───────────────────────────────────────────────────────
import sklearn
REQUIRED = "1.6.1"
if sklearn.__version__ != REQUIRED:
    print("\n" + "="*60)
    print(f"  ⚠  WARNING: sklearn version mismatch!")
    print(f"     Models were trained on sklearn {REQUIRED}")
    print(f"     You are running sklearn {sklearn.__version__}")
    print(f"     Predictions WILL be wrong (model predicts only 1-2 classes).")
    print(f"\n  Fix: run  fix_and_start.bat  (Windows)")
    print(f"       or   fix_and_start.sh   (Mac/Linux)")
    print(f"       to create the correct virtual environment.")
    print("="*60 + "\n")

# ── paths ──────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(__file__)
MODELS = os.path.join(BASE, "models")

# ── load artifacts once at startup ─────────────────────────────────────────────
le_dict    = joblib.load(os.path.join(MODELS, "label_encoders.pkl"))   # dict of LabelEncoders
le_target  = joblib.load(os.path.join(MODELS, "le_target.pkl"))        # LabelEncoder for y
scaler     = joblib.load(os.path.join(MODELS, "scaler.pkl"))           # StandardScaler
model      = joblib.load(os.path.join(MODELS, "best_model.pkl"))       # GradientBoostingClassifier

# ── feature order must match training exactly ───────────────────────────────────
FEATURE_ORDER = [
    "gender", "sexual_orientation", "location_type", "income_bracket",
    "education_level", "app_usage_time_min", "swipe_right_ratio",
    "likes_received", "mutual_matches", "profile_pics_count", "bio_length",
    "message_sent_count", "emoji_usage_rate", "last_active_hour",
    "swipe_time_of_day",
]

CATEGORICAL = set(le_dict.keys())  # {'gender','sexual_orientation','location_type',
                                    #  'income_bracket','education_level','swipe_time_of_day'}

def normalise_apostrophes(s: str) -> str:
    """Replace straight apostrophes with curly ones so they match the LabelEncoder classes."""
    return s.replace("'", "\u2019")

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)   # allow the frontend (any origin) to call this API



@app.get("/health")
def health():
    ok = sklearn.__version__ == REQUIRED
    return jsonify({"status": "ok" if ok else "version_mismatch","sklearn_installed": sklearn.__version__,"sklearn_required": REQUIRED,"version_ok": ok}), 200

@app.get("/metadata")
def metadata():
    """Return all dropdown options and numeric ranges for the frontend."""
    return jsonify({
        "categorical_options": {
            col: encoder.classes_.tolist()
            for col, encoder in le_dict.items()
        },
        "outcome_classes": le_target.classes_.tolist(),
        "numeric_features": [f for f in FEATURE_ORDER if f not in CATEGORICAL],
        "feature_order": FEATURE_ORDER,
    })


@app.post("/predict")
def predict():
    """
    Expects JSON body with all 15 features, e.g.:
    {
      "gender": "Male",
      "sexual_orientation": "Straight",
      "location_type": "Urban",
      "income_bracket": "High",
      "education_level": "Bachelor\u2019s",
      "app_usage_time_min": 52,
      "swipe_right_ratio": 0.6,
      "likes_received": 173,
      "mutual_matches": 23,
      "profile_pics_count": 4,
      "bio_length": 44,
      "message_sent_count": 75,
      "emoji_usage_rate": 0.36,
      "last_active_hour": 13,
      "swipe_time_of_day": "Early Morning"
    }
    """
    data = request.get_json(force=True)

    # Validate all features present
    missing = [f for f in FEATURE_ORDER if f not in data]
    if missing:
        return jsonify({"error": f"Missing features: {missing}"}), 400

    # Build feature vector
    row = []
    for feat in FEATURE_ORDER:
        val = data[feat]
        if feat in CATEGORICAL:
            val_str = normalise_apostrophes(str(val))
            try:
                val = float(le_dict[feat].transform([val_str])[0])
            except ValueError as e:
                known = le_dict[feat].classes_.tolist()
                return jsonify({"error": f"Unknown value '{val}' for '{feat}'. Known: {known}"}), 400
        else:
            try:
                val = float(val)
            except (TypeError, ValueError):
                return jsonify({"error": f"Feature '{feat}' must be numeric, got: {val}"}), 400
        row.append(val)

    # Scale and predict
    X = np.array(row, dtype=float).reshape(1, -1)
    X_scaled = scaler.transform(X)
    pred_enc  = model.predict(X_scaled)[0]
    proba     = model.predict_proba(X_scaled)[0]

    outcome = le_target.inverse_transform([pred_enc])[0]
    prob_map = {cls: round(float(p), 4) for cls, p in zip(le_target.classes_, proba)}

    return jsonify({
        "prediction": outcome,
        "confidence": round(float(proba[pred_enc]), 4),
        "probabilities": prob_map,
    })


if __name__ == "__main__":
    print("DateMatch API running at http://localhost:5000")
    app.run(debug=True, port=5000)