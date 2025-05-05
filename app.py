from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and encoder
model = joblib.load('model.pkl')
encoder = joblib.load('encoder.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    symptoms = ['fever','headache','nausea','vomiting','fatigue',
                'joint_pain','skin_rash','cough','weight_loss','yellow_eyes']
    try:
        values = [int(request.form[s]) for s in symptoms]
        prediction = model.predict([values])
        disease = encoder.inverse_transform(prediction)[0]
        return render_template('index.html', result=f"Predicted Disease: {disease}")
    except:
        return render_template('index.html', result="Invalid input!")

if __name__ == '__main__':
    app.run(debug=True)
