import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scr.data_management import load_full_cleaned_data

sns.set_style("whitegrid")

# Load dataset (update path if needed)
df = load_full_cleaned_data()


def page_data_study_body():
    st.write("### 🧪 Data Study: Feature Influence on Sale Price")

    # Context
    st.info(
        f"📌 **Business Requirement 1**\n\n"
        f"The client wants to understand how house attributes influence sale price.\n"
        f"This section provides correlation insights and visualisations based on the Ames Housing dataset."
    )

    # Load data
    df = load_full_cleaned_data()
    
    # Inspect dataset
    if st.checkbox("🔍 Inspect Dataset (first 10 rows)"):
        st.dataframe(df.head(10))
        st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    st.write("---")
    st.markdown("#### 🔗 Correlation Study")

    # Correlation summary
    st.write(
        f"* We conducted a correlation study to evaluate how house size, quality, and age relate to sale price.\n"
        f"* The strongest positive correlations found were with: `OverallQual`, `GrLivArea`, `GarageArea`, `TotalBsmtSF`."
    )

    # Heatmap toggle
    if st.checkbox("Show Correlation Heatmap"):
        top_features = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageArea', 'TotalBsmtSF', 'YearBuilt', 'YearRemodAdd']
        corr = df[top_features].corr()
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        st.pyplot(fig)

    # Plot distributions of key features
    if st.checkbox("Show Feature Distributions"):
        features = ['GrLivArea', 'GarageArea', 'TotalBsmtSF', 'OverallQual']
        for feature in features:
            fig, ax = plt.subplots()
            sns.histplot(df[feature], kde=True, ax=ax)
            ax.set_title(f"{feature} Distribution")
            st.pyplot(fig)

    # Scatter: SalePrice vs GrLivArea
    if st.checkbox("Show SalePrice vs GrLivArea"):
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x='GrLivArea', y='SalePrice', hue='OverallQual', palette='viridis')
        ax.set_title("Sale Price vs GrLivArea")
        st.pyplot(fig)

    st.success("""
    **Conclusions:**
    - Larger homes with higher material quality tend to sell for significantly more.
    - `OverallQual` is the top predictor, followed closely by `GrLivArea`.
    - Moderate correlations were observed for year built and renovations.
    
    These insights shaped our feature selection for the ML model.
    """)