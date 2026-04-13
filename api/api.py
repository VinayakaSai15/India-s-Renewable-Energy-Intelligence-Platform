from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)

# Example function (replace with your real models)
def run_prediction(data):
    # Example logic
    demand = data.get("demand", 0)
    solar = data.get("solar", 0)

    prediction = demand + solar  # dummy logic
    return prediction

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    result = run_prediction(data)

    return jsonify({
        "prediction": result
    })

# Required for Vercel
def handler(request):
    return app(request)