import streamlit as st
import pandas as pd
import numpy as np
from scr.data_management import load_pkl_file, load_inherited_houses


def page_house_price_predictor_body():
    st.write("### 💰 House Price Predictor")

    st.info(
        "Input the key property features below to predict sale price. "
        "This supports Business Requirement 2."
    )

    # Load simplified model and selected features
    model = load_pkl_file(
        "outputs/ml_pipeline/predict_price/v1/regression_pipeline_simple.pkl")
    selected_features = [
        'GrLivArea', 'OverallQual', 'GarageArea', 'TotalBsmtSF', 'YearBuilt']

    # Load and filter inherited houses
    inherited_df = load_inherited_houses()[selected_features]

    # Dropdown to select one of Lydia's inherited houses
    house_names = [f"Lydia's House {i+1}" for i in range(len(inherited_df))]
    selected_house = st.selectbox(
        "Or load one of Lydia's inherited houses:", [
            "Select a house..."] + house_names)

    # Determine defaults based on dropdown
    defaults = inherited_df.iloc[house_names.index(
        selected_house)] if selected_house != "Select a house..." else None

    # Create form input fields
    col1, col2 = st.columns(2)
    X_live = pd.DataFrame(index=[0])

    X_live['GrLivArea'] = col1.number_input(
        "Above Grade Living Area (sq ft)", 300, 6000,
        int(defaults['GrLivArea']) if defaults is not None else 1500
    )
    X_live['OverallQual'] = col2.slider(
        "Overall Quality (1–10)", 1, 10,
        int(defaults['OverallQual']) if defaults is not None else 6
    )
    X_live['GarageArea'] = col1.number_input(
        "Garage Area (sq ft)", 0, 1500,
        int(defaults['GarageArea']) if defaults is not None else 400
    )
    X_live['TotalBsmtSF'] = col2.number_input(
        "Total Basement SF", 0, 3000,
        int(defaults['TotalBsmtSF']) if defaults is not None else 900
    )
    X_live['YearBuilt'] = col1.number_input(
        "Year Built", 1870, 2025,
        int(defaults['YearBuilt']) if defaults is not None else 1995
    )

    # Predict and display
    if st.button("💡 Predict Sale Price"):
        sale_price = model.predict(X_live)[0]
        st.success(f"🏡 Estimated Sale Price: **${sale_price:,.0f}**")
        