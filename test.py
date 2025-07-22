import requests

#  make a call api to  flask app predict endpoint 

def call_predict_api(data):
    
    url = 'http://127.0.0.1:5000/predict'

    response = requests.post(url, json=data)
    if response.status_code == 200: 
        return response.json()
    else:
        raise Exception(f"API call failed with status code {response.status_code}")

if __name__ == "__main__":
    try:
        # create a data dictionary to send in the request

        data = {
            'avg_rooms_per_dwelling': 6.5,          # Average number of rooms per dwelling
            'lower_status_pct': 12.3,               # % of population considered lower status
            'distance_to_employment': 4.2,          # Weighted distance to employment centers
            'property_tax_rate': 330.0,             # Property tax rate per $10,000
            'crime_rate': 0.1                       # Per capita crime rate
        }
    
        prediction = call_predict_api(data)
        print(f"{prediction}")
    except Exception as e:
        print(f"Error occurred: {e}")