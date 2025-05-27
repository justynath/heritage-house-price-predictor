import streamlit as st
import pandas as pd
from scr.data_management import load_pkl_file
from scr.machine_learning.predict_price import predict_sale_price  # optional

def page_house_price_predictor_body():
    st.write("### 💰 House Price Predictor")

    st.info(
        "Input house features below to predict the sale price. This supports Business Requirement 2."
    )

    # Load the trained simplified pipeline
    model = load_pkl_file("outputs/ml_pipeline/predict_price/v1/regression_pipeline.pkl")

    # Build input form
    X_live = pd.DataFrame([], index=[0])
    col1, col2 = st.columns(2)

    X_live['GrLivArea'] = col1.number_input("Above Grade Living Area", 300, 6000, 1500)
    X_live['OverallQual'] = col2.slider("Overall Quality (1–10)", 1, 10, 6)
    X_live['TotalBsmtSF'] = col1.number_input("Total Basement SF", 1, 3000, 900)
    X_live['GarageArea'] = col2.number_input("Garage Area", 1, 1500, 400)
    X_live['YearBuilt'] = col1.number_input("Year Built", 1870, 2023, 1995)

    # Prediction
    if st.button("💡 Predict Sale Price"):
        # Align with model's expected feature names and order
        expected_cols = model.feature_names_in_

        for col in expected_cols:
            if col not in X_live.columns:
                X_live[col] = 0  # or np.nan or any safe default

        X_live = X_live[expected_cols]

        sale_price = model.predict(X_live)[0]
        st.success(f"🏡 Estimated Sale Price: **${sale_price:,.0f}**")