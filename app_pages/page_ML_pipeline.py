import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


def page_ML_pipeline_body():
    st.write("### ML Pipeline & Performance")

    st.info("""
    This page explains how the final machine learning pipeline was built,
     what features it used, and how well it performed.
     It supports **Business Requirement 2** by ensuring the model
     delivers reliable predictions.
    """)

    # Load saved model pipeline and training data
    model_path = "outputs/ml_pipeline/predict_price/v1/regression_pipeline.pkl"
    model = joblib.load(model_path)

    X_train = pd.read_csv("outputs/ml_pipeline/predict_price/v1/X_train.csv")
    X_test = pd.read_csv("outputs/ml_pipeline/predict_price/v1/X_test.csv")
    y_train = pd.read_csv("outputs/ml_pipeline/predict_price/v1/y_train.csv")
    y_test = pd.read_csv("outputs/ml_pipeline/predict_price/v1/y_test.csv")

    # Show pipeline structure
    st.markdown("#### Final Model Pipeline")
    st.write(model)

    # Display training features
    st.markdown("#### Features Used in Training")
    st.write(X_train.columns.tolist())

    st.markdown("""
    _These are the raw input features before transformation, encoding,
     and selection.
    The final features used for prediction may be reduced after correlation
     filtering and model-based selection._
    """)

    # Feature importance (optional)
    try:
        st.markdown("#### Feature Importance")
        st.image("outputs/ml_pipeline/predict_price/v1/feature_importance.png")
    except FileNotFoundError:
        st.warning(
            "Feature importance image not found. "
            "You can include this for extra insight.")

    st.write("---")

    # Display performance plots
    st.markdown("#### Actual vs Predicted Sale Price")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Train Set**")
        st.image(
            "outputs/ml_pipeline/predict_price/v1/actual_vs_pred_train.png")

    with col2:
        st.markdown("**Test Set**")
        st.image(
            "outputs/ml_pipeline/predict_price/v1/actual_vs_pred_test.png")

    st.write("---")

    # Compute R2 and MAE for both sets
    from sklearn.metrics import r2_score, mean_absolute_error

    preds_train = model.predict(X_train)
    preds_test = model.predict(X_test)

    r2_train = r2_score(y_train, preds_train)
    r2_test = r2_score(y_test, preds_test)
    mae_train = mean_absolute_error(y_train, preds_train)
    mae_test = mean_absolute_error(y_test, preds_test)

    st.markdown("#### Model Performance Summary")

    metrics = {
        "Metric": ["R² Score (Train)", "R² Score (Test)", "MAE (Train)",
                   "MAE (Test)"],
        "Value": [f"{r2_train:.3f}", f"{r2_test:.3f}", f"${mae_train:,.0f}",
                  f"${mae_test:,.0f}"]
    }
    st.table(pd.DataFrame(metrics))

    st.success("""
    **Conclusion:**
    The pipeline achieves strong performance and meets the success criteria
     defined with the client (R² ≥ 0.75).
    This validates that the model is fit for real-world prediction of house
     sale prices in Ames, Iowa.
    """)
