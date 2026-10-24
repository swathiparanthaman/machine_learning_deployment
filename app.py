
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load the newly trained Decision Tree model
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
"waste_decision_tree (1).pkl")

model = joblib.load(MODEL_PATH)

FEATURES = ["U1", "U2", "U3", "U4", "U5"]

# Internal labels used during model training
CLASS_NAMES = {
    0: "Low Waste",
    1: "Medium Waste",
    2: "High Waste"
}


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Smart Waste Management Decision Tree API is running"
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Send JSON sensor readings"
        }), 400

    try:
        values = [float(data[name]) for name in FEATURES]

        if not all(0 < value <= 28 for value in values):
            return jsonify({
                "error": "Sensor readings must be greater than 0 and at most 28 cm"
            }), 400

        inputs = pd.DataFrame(
            [values],
            columns=FEATURES
        )

        prediction = int(model.predict(inputs)[0])

        return jsonify({
            "prediction": prediction,
            "status": CLASS_NAMES[prediction]
        })

    except (KeyError, TypeError, ValueError) as error:
        return jsonify({
            "error": str(error)
        }), 400


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )
