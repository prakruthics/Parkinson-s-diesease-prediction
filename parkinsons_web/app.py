import numpy as np
import pickle
from flask import Flask, render_template, request

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get input values from the form
    features = [float(request.form[f'f{i}']) for i in range(7)]  # Assumes 7 features
    features = np.array(features).reshape(1, -1)
    
    # Apply the same scaling used during training
    scaled_features = scaler.transform(features)
    
    # Make prediction
    prediction = model.predict(scaled_features)
    
    # Map the prediction to a readable result
    result = "Parkinson's Disease detected" if prediction[0] == 1 else "No Parkinson's Disease"
    
    # Pass the result to the template
    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
