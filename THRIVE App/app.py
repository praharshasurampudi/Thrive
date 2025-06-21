from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('best_random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract inputs from the form
        input_data = [
            float(request.form['back_x']),
            float(request.form['back_y']),
            float(request.form['back_z']),
            float(request.form['thigh_x']),
            float(request.form['thigh_y']),
            float(request.form['thigh_z']),
        ]

        # Convert to 2D array
        features = np.array(input_data).reshape(1, -1)

        # Predict
        prediction = model.predict(features)

        return render_template('index.html', prediction_text=f'Predicted Label: {prediction[0]}')

    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
