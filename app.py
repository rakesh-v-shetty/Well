import pandas as pd
from flask import Flask, request, jsonify, render_template, redirect, url_for
import pickle
import numpy as np
import os

app = Flask(__name__)

# --- Load the Model and Auxiliary Data ---
MODEL_PATH = 'well_suitability_model.pkl'
FEATURES_PATH = 'model_features.pkl'
CATEGORICAL_OPTIONS_PATH = 'categorical_options.pkl'
LOCATION_COUNTS_PATH = 'location_counts_lookup.pkl'

model = None
model_features = None
categorical_options_for_template = {}
location_counts_lookup = pd.DataFrame()
average_regional_well_density = 100 # Default if lookup fails

try:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as file:
            model = pickle.load(file)
        print(f"Model loaded successfully from {MODEL_PATH}")
    else:
        print(f"Warning: Model file '{MODEL_PATH}' not found.")

    if os.path.exists(FEATURES_PATH):
        with open(FEATURES_PATH, 'rb') as file:
            model_features = pickle.load(file)
        print(f"Model features loaded successfully from {FEATURES_PATH}")
    else:
        print(f"Warning: Features file '{FEATURES_PATH}' not found.")

    if os.path.exists(CATEGORICAL_OPTIONS_PATH):
        with open(CATEGORICAL_OPTIONS_PATH, 'rb') as file:
            full_categorical_options = pickle.load(file)
            categorical_options_for_template['Lithology_Top_Layer'] = full_categorical_options.get('Lithology_Top_Layer', [])
            categorical_options_for_template['Drilling_Method_Used'] = full_categorical_options.get('Drilling_Method_Used', [])
            print(f"Filtered categorical options loaded successfully from {CATEGORICAL_OPTIONS_PATH}")
    else:
        print(f"Warning: Categorical options file '{CATEGORICAL_OPTIONS_PATH}' not found.")

    if os.path.exists(LOCATION_COUNTS_PATH):
        with open(LOCATION_COUNTS_PATH, 'rb') as file:
            location_counts_lookup = pickle.load(file)
        print(f"Location counts lookup loaded successfully from {LOCATION_COUNTS_PATH}")
        if not location_counts_lookup.empty:
            average_regional_well_density = location_counts_lookup['Regional_Well_Density_Count'].mean()
            print(f"Calculated average regional well density: {average_regional_well_density:.2f}")
    else:
        print(f"Warning: Location counts lookup file '{LOCATION_COUNTS_PATH}' not found. Using default average density.")

    if model is None or model_features is None:
        print("CRITICAL ERROR: Model or features could not be loaded. Server will not function correctly for predictions.")

except Exception as e:
    print(f"An error occurred during model/auxiliary data loading: {e}")
    model = None
    model_features = None

# Define expected input features from the form
EXPECTED_INPUT_FEATURES = [
    'Name of State',
    'Name of District',
    'Lithology_Top_Layer',
    'Drilling_Method_Used',
    'Well_Depth_Actual_m',
    'Well_Discharge_LPM',
    'Water_Quality_TDS_mg_L',
    'Total Annual Ground Water Recharge',
    'Net Ground Water Availability for future use',
    'Stage of Ground Water Extraction (%)'
]

@app.route('/')
def home():
    """Renders the main home page."""
    return render_template('home.html')

@app.route('/prediction_form') # <--- THIS ROUTE IS CRUCIAL
def show_prediction_form():
    """Renders the prediction form page."""
    return render_template('prediction_form.html', # <--- THIS FILENAME IS CRUCIAL
                           input_features=EXPECTED_INPUT_FEATURES,
                           categorical_options=categorical_options_for_template)

@app.route('/predict', methods=['POST'])
def predict():
    """Handles prediction requests."""
    if model is None or model_features is None:
        error_message = 'Model or features not loaded. Server not ready for predictions.'
        return render_template('prediction_form.html',
                               prediction_error=error_message,
                               input_features=EXPECTED_INPUT_FEATURES,
                               categorical_options=categorical_options_for_template)


    try:
        data = request.form.to_dict()
        print(f"Received data: {data}")

        for feature in EXPECTED_INPUT_FEATURES:
            if feature in ['Well_Depth_Actual_m', 'Well_Discharge_LPM',
                           'Water_Quality_TDS_mg_L', 'Total Annual Ground Water Recharge',
                           'Net Ground Water Availability for future use', 'Stage of Ground Water Extraction (%)']:
                try:
                    data[feature] = float(data[feature])
                except ValueError:
                    raise ValueError(f"'{feature.replace('_', ' ').title()}' must be a valid number. Received: '{data[feature]}'")
            elif feature in ['Name of State', 'Name of District', 'Lithology_Top_Layer', 'Drilling_Method_Used']:
                data[feature] = str(data[feature]).strip()
                if feature in ['Name of State', 'Name of District']:
                    data[feature] = data[feature].title()

        input_df_base = pd.DataFrame([data])

        input_df_base['Latitude'] = 22.0
        input_df_base['Longitude'] = 78.0

        engineered_df = input_df_base.copy()

        engineered_df['Depth_to_Discharge_Ratio'] = engineered_df['Well_Depth_Actual_m'] / (engineered_df['Well_Discharge_LPM'] + 1)

        engineered_df['Water_Quality_Category'] = pd.cut(engineered_df['Water_Quality_TDS_mg_L'],
                                                          bins=[0, 500, 1000, 1500, float('inf')],
                                                          labels=['Excellent', 'Good', 'Fair', 'Poor'],
                                                          right=True, include_lowest=True)

        engineered_df['Groundwater_Stress'] = engineered_df['Stage of Ground Water Extraction (%)'] / 100
        engineered_df['Recharge_Availability_Ratio'] = (engineered_df['Total Annual Ground Water Recharge'] /
                                                          (engineered_df['Net Ground Water Availability for future use'] + 1))

        def get_regional_density(row):
            state = row['Name of State']
            district = row['Name of District']
            density_row = location_counts_lookup[(location_counts_lookup['Name of State'] == state) &
                                                 (location_counts_lookup['Name of District'] == district)]
            if not density_row.empty:
                return density_row['Regional_Well_Density_Count'].iloc[0]
            else:
                print(f"Warning: State/District combination '{state}, {district}' not found in training data. Using average density: {average_regional_well_density:.2f}")
                return average_regional_well_density

        engineered_df['Regional_Well_Density'] = engineered_df.apply(get_regional_density, axis=1)

        final_input_df = engineered_df.reindex(columns=model_features, fill_value=np.nan)

        prediction = model.predict(final_input_df)[0]
        prediction_proba = model.predict_proba(final_input_df)[0].tolist()

        result = {
            'prediction': int(prediction),
            'suitability_status': 'Suitable' if prediction == 1 else 'Not Suitable',
            'probability_not_suitable': round(prediction_proba[0], 4),
            'probability_suitable': round(prediction_proba[1], 4)
        }
        print(f"Prediction result: {result}")
        return render_template('prediction_form.html', prediction_result=result,
                               input_data=data,
                               input_features=EXPECTED_INPUT_FEATURES,
                               categorical_options=categorical_options_for_template)

    except ValueError as ve:
        error_message = f"Input Error: {ve}"
        print(f"Validation Error: {error_message}")
        return render_template('prediction_form.html', prediction_error=error_message,
                               input_data=request.form.to_dict(),
                               input_features=EXPECTED_INPUT_FEATURES,
                               categorical_options=categorical_options_for_template)
    except Exception as e:
        error_message = f"An unexpected error occurred during prediction: {e}"
        print(f"Prediction Error: {error_message}")
        return render_template('prediction_form.html', prediction_error=error_message,
                               input_data=request.form.to_dict(),
                               input_features=EXPECTED_INPUT_FEATURES,
                               categorical_options=categorical_options_for_template)

if __name__ == '__main__':
    if not all(os.path.exists(p) for p in [MODEL_PATH, FEATURES_PATH, CATEGORICAL_OPTIONS_PATH, LOCATION_COUNTS_PATH]):
        print("\n!!! WARNING: One or more model-related files are missing. Please run the training script first. !!!\n")
        print("The Flask app will start, but predictions might not work correctly or at all.")
    app.run(debug=True)