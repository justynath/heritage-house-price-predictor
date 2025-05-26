import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scr.data_management import load_full_cleaned_data

def page_hypothesis_and_validation_body():
    st.write("### Hypotheses and Validation")

    st.info("""
    This page presents the project's hypotheses, how they were tested, and what the data revealed. These insights shaped our model design and feature selection.
    """)

    # Load cleaned dataset
    df = load_full_cleaned_data()

    # Hypotheses summary
    st.markdown("#### Project Hypotheses")
    st.markdown("""
    1. **Larger houses tend to sell for higher prices**  
       → Investigated using `GrLivArea`, `TotalBsmtSF`, and `GarageArea`.

    2. **Higher quality homes are more valuable**  
       → Investigated using `OverallQual` and `KitchenQual`.

    3. **Newer or recently renovated houses sell for more**  
       → Investigated using `YearBuilt` and `YearRemodAdd`.
    """)

    # Table summarising results
    st.markdown("#### Validation Summary")

    data = {
        "Hypothesis": [
            "Larger homes sell for more",
            "Higher quality increases value",
            "Newer/renovated homes are more expensive"
        ],
        "Outcome": [
            "Supported",
            "Supported",
            "Partially supported"
        ],
        "Validation": [
            "Strong correlation with `GrLivArea`, `GarageArea`, `TotalBsmtSF`",
            "`OverallQual` and `KitchenQual` are top predictors",
            "Moderate correlation with `YearBuilt` and `YearRemodAdd`"
        ]
    }
    table = pd.DataFrame(data, index=["1", "2", "3"])
    st.dataframe(table)

    # Optional: Correlation heatmap
    if st.checkbox("Show correlation heatmap of key features"):
        top_vars = ['SalePrice', 'GrLivArea', 'OverallQual', 'GarageArea', 'TotalBsmtSF', 'KitchenQual', 'YearBuilt', 'YearRemodAdd']
        # Drop non-numeric columns (e.g., KitchenQual is likely a string)
        numeric_subset = df[top_vars].select_dtypes(include='number')
        corr = numeric_subset.corr()

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        st.pyplot(fig)

    st.success("""
    **Conclusion:**  
    These findings confirmed our assumptions and guided our model to focus on size, quality, and structural age — improving both interpretability and predictive performance.
    """)