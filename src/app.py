from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model and scaler
with open("/workspaces/bertuzzi-machine-learning-final/src/rf_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("/workspaces/bertuzzi-machine-learning-final/src/scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route("/")
def home():
    return render_template("form.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract and convert form inputs
        features = [float(request.form.get(f)) for f in [
            'calories', 'protein', 'carbohydrates', 'fat', 
            'high_protein', 'low_carb', 'is_balanced'
        ]]

        # Scale the input
        input_scaled = scaler.transform([[
            'calories', 'protein', 'carbohydrates', 'fat'
        ]])

        # Predict using the model
        prediction = model.predict(input_scaled)[0]

        return render_template("form.html", prediction_text=f"Predicted Health Score: {int(prediction)}")

    except Exception as e:
        return render_template("form.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
