import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scr.data_management import load_full_cleaned_data


def page_hypothesis_and_validation_body():
    st.write("### Hypotheses and Validation")

    st.info("""
    This page presents the project's hypotheses, how they were tested,
    what the data revealed, and how those insights influenced
     the machine learning model design.
    """)

    # Load cleaned dataset
    df = load_full_cleaned_data()

    # Section: Hypotheses
    st.markdown("#### Project Hypotheses")

    st.markdown("""
    **1. Larger houses tend to sell for higher prices**
    - **Validation Method**: Pearson correlation
    - **Key Features**: `GrLivArea`, `TotalBsmtSF`, `GarageArea`
    - **Findings**:
        - `GrLivArea` had a strong positive correlation with
         `SalePrice` (**0.71**)
        - `GarageArea` (**0.64**) and `TotalBsmtSF` (**0.61**)
         also had strong correlations
    - **Conclusion**: Hypothesis strongly supported.
    - **Action Taken**: These features were prioritised in the modelling phase.
     Lydia was advised to highlight these aspects when pricing or marketing.

    **2. Higher quality homes are more valuable**
    - **Validation Method**: Pearson correlation
    - **Key Features**: `OverallQual`, `KitchenQual`
    - **Findings**:
        - `OverallQual` had the strongest correlation with
         `SalePrice` (**0.79**)
        - `KitchenQual` was also strongly correlated (**0.67**)
    - **Conclusion**: Hypothesis strongly supported.
    - **Action Taken**: These features were given higher importance
     in feature selection. Lydia could consider renovations
     to improve perceived quality.

    **3. Newer or recently renovated homes sell for more**
    - **Validation Method**: Pearson correlation
    - **Key Features**: `YearBuilt`, `YearRemodAdd`
    - **Findings**:
        - `YearBuilt` (**0.52**) and `YearRemodAdd` (**0.51**)
         had moderate positive correlations
    - **Conclusion**: Partially supported.
    - **Action Taken**: These variables were included in the model. Listings
     were advised to emphasise renovation history and structural updates.
    """)

    # Optional table summary
    st.markdown("#### Validation Summary Table")
    data = {
        "Hypothesis": [
            "Larger homes sell for more",
            "Higher quality increases value",
            "Newer/renovated homes are more expensive"
        ],
        "Outcome": [
            "Supported (r = 0.71–0.61)",
            "Supported (r = 0.79, 0.67)",
            "Partially supported (r = 0.52, 0.51)"
        ],
        "Validation": [
            "Strong correlation with GrLivArea, GarageArea, TotalBsmtSF",
            "OverallQual and KitchenQual are top predictors",
            "Moderate correlation with YearBuilt and YearRemodAdd"
        ]
    }
    table = pd.DataFrame(data, index=["1", "2", "3"])
    st.dataframe(table)

    # Optional correlation heatmap
    if st.checkbox("Show correlation heatmap of key features"):
        top_vars = ['SalePrice', 'GrLivArea', 'OverallQual', 'GarageArea',
                    'TotalBsmtSF', 'KitchenQual', 'YearBuilt', 'YearRemodAdd']
        numeric_subset = df[top_vars].select_dtypes(include='number')
        corr = numeric_subset.corr()

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        st.pyplot(fig)

    st.success("""
    **Conclusion:**
    These findings confirmed key assumptions and shaped the model to focus
     on size, quality, and structural age. This helped improve both the
     interpretability and performance of the machine learning solution,
     and gave Lydia clear criteria for pricing and marketing.
    """)
    