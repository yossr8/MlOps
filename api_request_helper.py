import pandas as pd
import requests
import json

def prepare_titanic_features(passenger_data):
    """
    Transform raw passenger data into the format expected by the model.
    
    Args:
        passenger_data: Dictionary with passenger details
        
    Returns:
        Dictionary formatted for the API with proper one-hot encoding
    """
    # Create a DataFrame with one row
    df = pd.DataFrame([passenger_data])
    
    # Handle categorical features with one-hot encoding
    # For Pclass
    df['Pclass_1'] = 0
    df['Pclass_2'] = 0
    df['Pclass_3'] = 0
    if 'Pclass' in df.columns:
        pclass_val = df['Pclass'].iloc[0]
        if pclass_val == 1:
            df['Pclass_1'] = 1
        elif pclass_val == 2:
            df['Pclass_2'] = 1
        elif pclass_val == 3:
            df['Pclass_3'] = 1
    
    # For Sex
    df['Sex_female'] = 0
    df['Sex_male'] = 0
    if 'Sex' in df.columns:
        sex_val = df['Sex'].iloc[0]
        if sex_val.lower() == 'female':
            df['Sex_female'] = 1
        elif sex_val.lower() == 'male':
            df['Sex_male'] = 1
    
    # For Embarked
    df['Embarked_C'] = 0
    df['Embarked_Q'] = 0
    df['Embarked_S'] = 0
    if 'Embarked' in df.columns:
        embarked_val = df['Embarked'].iloc[0]
        if embarked_val == 'C':
            df['Embarked_C'] = 1
        elif embarked_val == 'Q':
            df['Embarked_Q'] = 1
        elif embarked_val == 'S':
            df['Embarked_S'] = 1
    
    # Keep only the necessary columns (removing original categorical columns)
    # Based on the error message, these are the columns needed by the model
    columns_to_keep = [
        'Age', 'SibSp', 'Parch', 'Fare',
        'Pclass_1', 'Pclass_2', 'Pclass_3',
        'Sex_female', 'Sex_male',
        'Embarked_C', 'Embarked_Q', 'Embarked_S'
    ]
    
    # Only keep columns that exist in our DataFrame
    columns_to_keep = [col for col in columns_to_keep if col in df.columns]
    df = df[columns_to_keep]
    
    # Convert to dictionary for API request
    return df.to_dict(orient='records')[0]

# Example usage
passenger = {
    "Age": 29.0,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 7.25,
    "Pclass": 3,
    "Sex": "male",
    "Embarked": "S"
}

# Format the data for the API
formatted_data = prepare_titanic_features(passenger)
print("Properly formatted data for API request:")
print(json.dumps(formatted_data, indent=4))

# Example of how to send this to your API
# Replace 'YOUR_API_ENDPOINT' with the actual endpoint
def predict_survival(data, api_url="http://127.0.0.1:8000/predict"):
    response = requests.post(api_url, json=data)
    return response.json()

# Uncomment to use:
prediction = predict_survival(formatted_data)
print("Prediction:", prediction)