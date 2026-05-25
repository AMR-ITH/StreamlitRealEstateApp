import streamlit as st
import pandas as pd
import numpy as np
import joblib
import traceback

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="Bangalore Flats Price Prediction",
    page_icon="🏢"
)

# -----------------------------------
# LOAD PREPROCESSOR
# -----------------------------------
@st.cache_resource
def load_preprocessor():

    try:
        preprocessor = joblib.load("preprocessor.joblib")
        return preprocessor

    except Exception as e:
        st.error(f"Error loading preprocessor: {str(e)}")
        return None


# -----------------------------------
# TITLE
# -----------------------------------
st.title("Bangalore Real Estate Price Predictor")

st.markdown(
    "### Get estimated prices for flats in Bangalore"
)

# -----------------------------------
# INPUT SECTION
# -----------------------------------
col1, col2 = st.columns(2)

with col1:

    zone = st.selectbox(
        "Zone/Sector",
        ["east", "west", "north", "south"]
    )

    construction_status = st.selectbox(
        "Construction Status",
        [
            "New Property",
            "Under Construction",
            "Relatively New",
            "Moderatly Old",
            "Old",
            "undefined"
        ]
    )

    bhk = float(
        st.selectbox(
            "Number of Bedrooms",
            ["1", "2", "3", "4", "5", "6", "7"]
        )
    )

with col2:

    built_up_area = float(
        st.number_input(
            "Built Up Area (sq.ft.)",
            min_value=100.0,
            value=1000.0,
            step=100.0
        )
    )

    facility_category = st.selectbox(
        "Facility Category",
        ["low", "medium", "high"]
    )

# -----------------------------------
# DIVIDER
# -----------------------------------
st.markdown("---")

# -----------------------------------
# BUTTON
# -----------------------------------
predict_button = st.button(
    "Predict Price",
    type="primary"
)

# -----------------------------------
# PREDICTION
# -----------------------------------
if predict_button:

    with st.spinner("Predicting price..."):

        try:

            # -----------------------------
            # CREATE INPUT DATAFRAME
            # -----------------------------
            input_data = pd.DataFrame({

                "zone": [zone],

                "construction_status": [
                    construction_status
                ],

                "bhk_type": [bhk],

                "bulit_area": [built_up_area],

                "luxury_category": [
                    facility_category
                ]
            })

            # -----------------------------
            # LOAD PREPROCESSOR
            # -----------------------------
            preprocessor = load_preprocessor()

            if preprocessor is None:

                st.error(
                    "Unable to load preprocessor."
                )

                st.stop()

            # -----------------------------
            # TRANSFORM INPUT
            # -----------------------------
            preprocessed_input = preprocessor.transform(
                input_data
            )

            # -----------------------------------
            # SIMPLE DEMO PREDICTION LOGIC
            # -----------------------------------
            base_price = 5000

            zone_multipliers = {
                "east": 0.9,
                "west": 1.1,
                "north": 1.0,
                "south": 1.2
            }

            facility_multipliers = {
                "low": 0.85,
                "medium": 1.0,
                "high": 1.3
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

            # -----------------------------------
            # PRICE RANGE
            # -----------------------------------
            test_mae = predicted_price * 0.10

            lower_bound = max(
                0,
                predicted_price - test_mae
            )

            upper_bound = (
                predicted_price + test_mae
            )

            # -----------------------------------
            # DISPLAY OUTPUT
            # -----------------------------------
            st.success("Prediction Complete!")

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

            st.write("Detailed error information:")

            st.code(
                traceback.format_exc()
            )
