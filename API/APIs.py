from flask import Flask, request, jsonify
from flask_cors import CORS
from joblib import load
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
model = load("house_model.pkl")

@app.route('/predict', methods=['POST'])
def predict_price():
    try:
        data = request.get_json()
        num_rooms = data.get("num_rooms")
        area = data.get("area")
        
        if num_rooms is None or area is None:
            return jsonify({"error": "Missing required parameters"}), 400
        
        # Prepare features for prediction
        features = np.array([[area, num_rooms]])  # Adjusted feature order
        prediction = model.predict(features)[0][0] * 1_000_000  # Convert to full Rwandan Francs
        
        return jsonify({"predicted_price": f"{prediction:,.0f} Rwf"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
