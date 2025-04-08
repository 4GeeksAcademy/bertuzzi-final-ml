from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model and scaler
with open("/workspaces/bertuzzi-final-ml/src/rf_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("/workspaces/bertuzzi-final-ml/src/scaler.pkl", "rb") as scaler_file:
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

        # Scale only the first 4 features
        input_scaled = scaler.transform([features[:4]])

        # Combine scaled numeric and unscaled binary features
        final_input = np.concatenate([input_scaled[0], features[4:]])

        # Predict using the model
        prediction = model.predict([final_input])[0]

        return render_template("form.html", prediction_text=f"Predicted Health Score: {int(prediction)}")

    except Exception as e:
        return render_template("form.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
