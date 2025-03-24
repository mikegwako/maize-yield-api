 
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load("maize_yield_model.pkl")

@app.route('/')
def home():
    return "Maize Yield Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON input
        data = request.get_json()

        # Convert to DataFrame (Make sure feature names match your training data)
        df = pd.DataFrame([data])

        # Predict yield
        prediction = model.predict(df)

        return jsonify({"predicted_yield": prediction.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
