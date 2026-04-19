import os
import pickle
from flask import Flask, request, render_template
import numpy as np


application = Flask(__name__)
app = application

# Get the absolute path of the current file (application.py)
# Then dirname() removes the file name and gives the project folder path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Create full path to ridge model file
# Joins BASE_DIR + 'models' + 'ridge.pkl' safely
model_path = os.path.join(BASE_DIR, 'models', 'ridge.pkl')

# Create full path to scaler file
# Ensures correct path regardless of where the app is run
scaler_path = os.path.join(BASE_DIR, 'models', 'scaler.pkl')

# Load model and scaler correctly
ridge_model = pickle.load(open(model_path, 'rb'))
standard_scaler = pickle.load(open(scaler_path, 'rb'))

# Debug (remove later)
print("Loaded Model:", type(ridge_model))
print("Loaded Scaler:", type(standard_scaler))


@app.route("/")
def index():
    return render_template('home.html')


@app.route("/predictdata", methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == "POST":
        try:
            # Get inputs
            Temperature = float(request.form.get("Temperature"))
            RH = float(request.form.get("RH"))
            Ws = float(request.form.get("Ws"))
            Rain = float(request.form.get("Rain"))
            FFMC = float(request.form.get("FFMC"))
            DMC = float(request.form.get("DMC"))
            ISI = float(request.form.get("ISI"))
            Classes = float(request.form.get("Classes"))
            Region = float(request.form.get("Region"))

            # Convert to array
            input_data = np.array([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])

            # Scale
            new_data_scaled = standard_scaler.transform(input_data)

            # Predict
            result = ridge_model.predict(new_data_scaled)

            return render_template('home.html', results=round(result[0], 2))

        except Exception as e:
            return f"Error occurred: {e}"

    return render_template('home.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)