import streamlit as st

def page_summary_body():
    st.write("### Quick Project Summary")

    # Definitions based on README terms
    st.info(
        f"**Project Terms & Jargon**\n"
        f"* A **feature** is a property attribute such as size, quality, or year built.\n"
        f"* A **target variable** is what we want to predict—in this case, `SalePrice`.\n"
        f"* **Feature engineering** involves transforming and selecting variables to improve model performance.\n"
        f"* A **regression model** estimates a continuous value like house price based on input features.\n\n"
        f"**Project Dataset**\n"
        f"* The dataset comes from Ames, Iowa, with ~1.5K residential housing records.\n"
        f"* It includes attributes such as square footage, garage size, kitchen quality, and year built.\n"
        f"* The dataset was sourced from [Kaggle](https://www.kaggle.com/codeinstitute/housing-prices-data) and represents houses built between 1872 and 2010."
    )

    # Link to GitHub README
    st.write(
        f"* For additional information, please visit the "
        f"[Project README file](https://github.com/justynath/heritage-house-price-predictor)."
    )

    # Business requirements summary
    st.success(
        f"The project has 2 business requirements:\n"
        f"* **BR1** – The client wants to understand which property features most influence sale price, "
        f"supported by interactive data visualisations.\n"
        f"* **BR2** – The client wants to predict the sale price of four inherited homes in Ames, "
        f"as well as other properties, using a machine learning regression model."
    )