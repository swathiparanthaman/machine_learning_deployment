from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

model = joblib.load("decision_tree_model (2).joblib")
features = ["bottom_cm", "middle_cm", "top_cm"]

@app.route("/")
def home():
    return jsonify({"message": "Smart Bin API is running"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Send JSON sensor readings"}), 400

    try:
        values = [[float(data[name]) for name in features]]
        inputs = pd.DataFrame(values, columns=features)
        level = int(model.predict(inputs)[0])

        names = {0: "Empty", 1: "Level 1", 2: "Level 2", 3: "Level 3"}
        return jsonify({"level": level, "status": names[level]})
    except (KeyError, ValueError) as error:
        return jsonify({"error": str(error)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
