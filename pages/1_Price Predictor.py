import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import os
from pathlib import Path
import mlflow
import traceback
import requests
from io import BytesIO

st.set_page_config(page_title="Bangalore Flats Price Prediction", page_icon="🏢")

# Configure MLflow - using environment variables for authentication
# These should be set in Streamlit's secrets management
def setup_mlflow():
    try:
        # Get credentials from Streamlit secrets or environment variables
        dagshub_username = os.environ.get("DAGSHUB_USERNAME", st.secrets.get("DAGSHUB_USERNAME", ""))
        dagshub_token = os.environ.get("DAGSHUB_TOKEN", st.secrets.get("DAGSHUB_TOKEN", ""))
        repo_owner = 'AMR-ITH'
        repo_name = 'RealEstateInsights'
        
        # Set MLflow tracking URI
        os.environ['MLFLOW_TRACKING_URI'] = f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"
        
        # Set credentials for MLflow
        if dagshub_username and dagshub_token:
            os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_username
            os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token
            return True
        else:
            st.warning("DagsHub credentials not found. Running in local mode.")
            return False
            
    except Exception as e:
        st.warning(f"MLflow setup issue: {str(e)}. Running in local mode.")
        return False

def load_model_information():
    try:
        # First try loading from github raw content
        url = f"https://raw.githubusercontent.com/AMR-ITH/RealEstateInsights/main/datasets/run_information.json"
        response = requests.get(url)
        if response.status_code == 200:
            return json.loads(response.text)
        else:
            # Fallback to local file if exists
            local_path = "datasets/run_information.json"
            if os.path.exists(local_path):
                with open(local_path) as f:
                    return json.load(f)
            else:
                # Default values if file isn't found
                return {"model_name": "bangalore_housing_model"}
    except Exception as e:
        st.warning(f"Unable to load model information: {str(e)}")
        return {"model_name": "bangalore_housing_model"}  # Default fallback

def load_preprocessor():
    try:
        # First try loading from github
        url = f"https://raw.githubusercontent.com/AMR-ITH/RealEstateInsights/main/models/preprocessor.joblib"
        response = requests.get(url)
        if response.status_code == 200:
            return joblib.load(BytesIO(response.content))
        else:
            # Fallback to local file
            local_path = "datasets/preprocessor.joblib"
            if os.path.exists(local_path):
                return joblib.load(local_path)
            else:
                raise FileNotFoundError("Preprocessor file not found")
    except Exception as e:
        st.error(f"Error loading preprocessor: {str(e)}")
        return None

# Set title with some styling
st.title("Bangalore Real Estate Price Predictor")
st.markdown("### Get estimated prices for flats in Bangalore")

# Initialize MLflow connection
mlflow_connected = setup_mlflow()
if mlflow_connected:
    st.success("Successfully connected to MLflow tracking server")
else:
    st.info("Running without MLflow connection. Some features may be limited.")

# Load model information
model_info = load_model_information()
model_name = model_info.get('model_name', 'bangalore_housing_model')

# Create two columns for the form inputs
col1, col2 = st.columns(2)

with col1:
    # sector
    zone = st.selectbox('Zone/Sector', ['east', 'west', 'north', 'south'])
    
    # construction status
    construction_status = st.selectbox('Construction Status', 
                                      ['New Property', 'Under Construction', 
                                       'Relatively New', 'Moderatly Old', 
                                       'Old', 'undefined'])
    
    # bhk
    bhk = float(st.selectbox('Number of Bedrooms', ['1', '2', '3', '4', '5', '6', '7']))

with col2:
    # built_up_area
    built_up_area = float(st.number_input('Built Up Area (sq.ft.)', 
                                         min_value=100.0, value=1000.0, step=100.0))
    
    # facility_category
    facility_category = st.selectbox('Facility Category', ['low', 'medium', 'high'])

# Add a divider
st.markdown("---")

# Create a prediction button
predict_button = st.button("Predict Price", type="primary")

if predict_button:
    # Show a spinner while predicting
    with st.spinner("Predicting price..."):
        try:
            # Create input dataframe for prediction
            input_data = pd.DataFrame({
                'zone': [zone],
                'construction_status': [construction_status],
                'bhk_type': [bhk],
                'bulit_area': [built_up_area],
                'luxury_category': [facility_category]
            })
            
            # Load the preprocessor
            preprocessor = load_preprocessor()
            if preprocessor is None:
                st.error("Unable to load preprocessor. Cannot continue with prediction.")
                st.stop()
                
            # Preprocess the input
            preprocessed_input = preprocessor.transform(input_data)

            # Default test_mae to use if we can't get it from MLflow
            default_test_mae = 0.5307577857825541
            
            if mlflow_connected:
                try:
                    # Try to load the model from MLflow
                    alias_name = "staging_latest"
                    model_uri = f"models:/{model_name}@{alias_name}"
                    model = mlflow.pyfunc.load_model(model_uri)
                    
                    # Get metrics from the run
                    model_info = mlflow.models.get_model_info(model_uri)
                    run_id = model_info.run_id
                    client = mlflow.tracking.MlflowClient()
                    run = client.get_run(run_id)
                    test_mae = run.data.metrics.get("test_mae", default_test_mae)
                    
                    # Make prediction using the MLflow model
                    prediction = model.predict(preprocessed_input)
                    
                except Exception as e:
                    st.warning(f"Error loading model from MLflow: {str(e)}")
                    st.warning("Attempting to use local model...")
                    # Fallback to local model logic here
                    raise Exception("Local model fallback not implemented")
            else:
                st.warning("MLflow connection not available. Using demonstration mode.")
                # Demonstration mode - simplified prediction based on inputs
                # This is a very basic fallback when MLflow is unavailable
                base_price = 5000  # Base price per sq ft
                zone_multipliers = {'east': 0.9, 'west': 1.1, 'north': 1.0, 'south': 1.2}
                facility_multipliers = {'low': 0.85, 'medium': 1.0, 'high': 1.3}
                bhk_adder = bhk * 100000  # Each bedroom adds value
                
                # Simple formula for demo purposes
                price = (base_price * built_up_area * zone_multipliers[zone] * 
                         facility_multipliers[facility_category]) + bhk_adder
                
                prediction = [price]
                test_mae = default_test_mae * price  # Proportional uncertainty
            
            # Display the prediction
            st.success("Prediction Complete!")
            
            # Get the predicted price
            predicted_price = prediction[0]
            
            # Calculate the lower and upper bounds based on the test MAE
            lower_bound = max(0, predicted_price - test_mae)  # Ensure non-negative
            upper_bound = predicted_price + test_mae
            
            # Display the prediction with range
            st.markdown(f"## Estimated Price: ₹{predicted_price:,.2f}")
            st.markdown(f"### Price Range: ₹{lower_bound:,.2f} - ₹{upper_bound:,.2f}")
            
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.write("Detailed error information:")
            st.code(traceback.format_exc())
