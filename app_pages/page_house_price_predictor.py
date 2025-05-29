import streamlit as st
import pandas as pd
from scr.data_management import load_pkl_file
import numpy as np

def page_house_price_predictor_body():
    st.write("### 💰 House Price Predictor")

    st.info(
        "Input the key property features below to predict sale price. "
        "This supports Business Requirement 2."
    )

    # Load simplified model
    model = load_pkl_file(
        "outputs/ml_pipeline/predict_price/v1/regression_pipeline_simple.pkl"
    )

    # Define the same features used in the model
    features = ['GrLivArea', 'OverallQual', 'GarageArea', 'TotalBsmtSF', 'YearBuilt']

    # Create input form
    col1, col2 = st.columns(2)
    X_live = pd.DataFrame(index=[0])

    X_live['GrLivArea'] = col1.number_input("Above Grade Living Area (sq ft)", 300, 6000, 1500)
    X_live['OverallQual'] = col2.slider("Overall Quality (1–10)", 1, 10, 6)
    X_live['GarageArea'] = col1.number_input("Garage Area (sq ft)", 0, 1500, 400)
    X_live['TotalBsmtSF'] = col2.number_input("Total Basement SF", 0, 3000, 900)
    X_live['YearBuilt'] = col1.number_input("Year Built", 1870, 2025, 1995)

    if st.button("💡 Predict Sale Price"):
        sale_price = model.predict(X_live)[0]
        st.success(f"🏡 Estimated Sale Price: **${sale_price:,.0f}**")