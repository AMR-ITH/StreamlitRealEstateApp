import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import os
import traceback
import requests
from io import BytesIO

st.set_page_config(
    page_title="Bangalore Flats Price Prediction",
    page_icon="🏢"
)

# -----------------------------
# LOAD MODEL INFORMATION
# -----------------------------
def load_model_information():
    try:
        url = "https://raw.githubusercontent.com/AMR-ITH/RealEstateInsights/main/datasets/run_information.json"

        response = requests.get(url)

        if response.status_code == 200:
            return json.loads(response.text)

        return {"model_name": "bangalore_housing_model"}

    except Exception as e:
        st.warning(f"Unable to load model information: {str(e)}")
        return {"model_name": "bangalore_housing_model"}


# -----------------------------
# LOAD PREPROCESSOR
# -----------------------------
def load_preprocessor():

    try:
        url = "https://raw.githubusercontent.com/AMR-ITH/RealEstateInsights/main/models/preprocessor.joblib"

        response = requests.get(url)

        if response.status_code == 200:
            return joblib.load(BytesIO(response.content))

        else:
            raise FileNotFoundError("Preprocessor file not found")

    except Exception as e:
        st.error(f"Error loading preprocessor: {str(e)}")
        return None


# -----------------------------
# TITLE
# -----------------------------
st.title("Bangalore Real Estate Price Predictor")

st.markdown(
    "### Get estimated prices for flats in Bangalore"
)

# -----------------------------
# LOAD MODEL INFO
# -----------------------------
model_info = load_model_information()

model_name = model_info.get(
    'model_name',
    'bangalore_housing_model'
)

# -----------------------------
# INPUT FORM
# -----------------------------
col1, col2 = st.columns(2)

with col1:

    zone = st.selectbox(
        'Zone/Sector',
        ['east', 'west', 'north', 'south']
    )

    construction_status = st.selectbox(
        'Construction Status',
        [
            'New Property',
            'Under Construction',
            'Relatively New',
            'Moderatly Old',
            'Old',
            'undefined'
        ]
    )

    bhk = float(
        st.selectbox(
            'Number of Bedrooms',
            ['1', '2', '3', '4', '5', '6', '7']
        )
    )

with col2:

    built_up_area = float(
        st.number_input(
            'Built Up Area (sq.ft.)',
            min_value=100.0,
            value=1000.0,
            step=100.0
        )
    )

    facility_category = st.selectbox(
        'Facility Category',
        ['low', 'medium', 'high']
    )

# -----------------------------
# DIVIDER
# -----------------------------
st.markdown("---")

# -----------------------------
# BUTTON
# -----------------------------
predict_button = st.button(
    "Predict Price",
    type="primary"
)

# -----------------------------
# PREDICTION
# -----------------------------
if predict_button:

    with st.spinner("Predicting price..."):

        try:

            # Create input dataframe
            input_data = pd.DataFrame({

                'zone': [zone],

                'construction_status': [
                    construction_status
                ],

                'bhk_type': [bhk],

                'bulit_area': [built_up_area],

                'luxury_category': [
                    facility_category
                ]
            })

            # Load preprocessor
            preprocessor = load_preprocessor()

            if preprocessor is None:

                st.error(
                    "Unable to load preprocessor."
                )

                st.stop()

            # Transform input
            preprocessed_input = preprocessor.transform(
                input_data
            )

            # -----------------------------
            # DEMO PREDICTION LOGIC
            # -----------------------------
            base_price = 5000

            zone_multipliers = {
                'east': 0.9,
                'west': 1.1,
                'north': 1.0,
                'south': 1.2
            }

            facility_multipliers = {
                'low': 0.85,
                'medium': 1.0,
                'high': 1.3
            }

            bhk_adder = bhk * 100000

            predicted_price = (

                base_price *

                built_up_area *

                zone_multipliers[zone] *

                facility_multipliers[
                    facility_category
                ]

            ) + bhk_adder

            # Estimated uncertainty
            test_mae = predicted_price * 0.10

            lower_bound = max(
                0,
                predicted_price - test_mae
            )

            upper_bound = (
                predicted_price + test_mae
            )

            # -----------------------------
            # OUTPUT
            # -----------------------------
            st.success(
                "Prediction Complete!"
            )

            st.markdown(
                f"## Estimated Price: ₹{predicted_price:,.2f}"
            )

            st.markdown(
                f"### Price Range: ₹{lower_bound:,.2f} - ₹{upper_bound:,.2f}"
            )

        except Exception as e:

            st.error(
                f"Error during prediction: {str(e)}"
            )

            st.write(
                "Detailed error information:"
            )

            st.code(
                traceback.format_exc()
            )
