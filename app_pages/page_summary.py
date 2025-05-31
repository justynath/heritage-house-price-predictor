import streamlit as st


def page_summary_body():
    st.write("### Quick Project Summary")

    # Definitions based on README terms
    st.info(
        f"**Project Terms & Jargon**\n"
        f"* A **feature** is a property attribute such as size, quality, "
        f"or year built.\n"
        f"* A **target variable** is what we want to predict—in this case, "
        f"`SalePrice`.\n"
        f"* **Feature engineering** involves transforming and selecting "
        f"variables to improve model performance.\n"
        f"* A **regression model** estimates a continuous value like "
        f"house price based on input features.\n\n"
        f"**Project Dataset**\n"
        f"* The dataset comes from Ames, Iowa, with ~1.5K residential "
        F"housing records.\n"
        f"* It includes attributes such as square footage, garage size, "
        f"kitchen quality, and year built.\n"
        f"* The dataset was sourced from "
        f"[Kaggle](https://www.kaggle.com/codeinstitute/housing-prices-data) "
        f"and represents houses built between 1872 and 2010."
    )

    # Link to GitHub README
    st.write(
        f"* For additional information, please visit the "
        f"[Project README file]"
        f"(https://github.com/justynath/heritage-house-price-predictor)."
    )

    # Business requirements summary
    st.success(
        f"The project has 2 business requirements:\n"
        f"* **BR1** – The client wants to understand which property features "
        f"most influence sale price, "
        f"supported by interactive data visualisations.\n"
        f"* **BR2** – The client wants to predict the sale price of "
        f"four inherited homes in Ames, "
        f"as well as other properties, using a machine learning "
        f"regression model."
    )
