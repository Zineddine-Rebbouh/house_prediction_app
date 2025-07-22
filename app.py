from flask import Flask, request, jsonify
import joblib
# Flask app to serve the pre-trained model for predictions 


app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('./models/boston_housing_model.pkl')
print("Model loaded successfully.")

# Home route
@app.route('/')
def home():
    return "Flask app is running!"

@app.route('/predict' , methods=['POST'])
def predict(): 
    print("Received request for prediction.")
    data = request.get_json(force=True)
    print(f"Input data: {data}")
    
    # check if all required fields are present
    required_fields = ['avg_rooms_per_dwelling', 'lower_status_pct', 'distance_to_employment', 'property_tax_rate', 'crime_rate']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'Missing required field: {field}'}), 400
        
    # Convert input data to the format expected by the model
    data = [
        data['avg_rooms_per_dwelling'],
        data['lower_status_pct'],
        data['distance_to_employment'],
        data['property_tax_rate'],
        data['crime_rate']
    ]
    print(f"Formatted input data: {data}")
    prediction = model.predict([data])
    print(f"Prediction: {prediction}")
    return jsonify({'prediction': prediction.tolist()})

if __name__ == "__main__":
    app.run(debug=True)
    
    
    
