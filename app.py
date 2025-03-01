from flask import Flask, request, jsonify, render_template
import os
import pickle
import numpy as np
import test 

app = Flask(__name__)

# Initialize model on startup
@app.before_request
def initialize():
    # Path to your dataset
    file_path = "./symptom_disease_dataset.csv"
    
    # Check if we have a saved model
    if os.path.exists('model.pkl') and os.path.exists('encoder.pkl'):
        print("Loading saved model and encoder...")
        # Load the model and encoder
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        # Set the global variables in test.py
        test.global_model = model
        test.global_label_encoder = label_encoder
    else:
        print("Training new model...")
        # Initialize model using function from test.py
        model, label_encoder = test.initialize_model(file_path)
        
        # Save model and encoder for future use
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
        with open('encoder.pkl', 'wb') as f:
            pickle.dump(label_encoder, f)
    
    print("Model initialized and ready for predictions!")

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction API
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.json
        symptoms = data['symptoms']
        
        # Convert to appropriate type (integers)
        symptoms = [int(s) for s in symptoms]
        
        # Make prediction using the function from test.py
        prediction = test.get_prediction(symptoms)
        
        # Return prediction
        return jsonify({
            'prediction': prediction,
            'success': True
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        })

if __name__ == "__main__":
    # The app will initialize the model on first request
    app.run(debug=True)