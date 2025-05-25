import streamlit as st
st.set_page_config(page_title="🏡 Heritage House Price Predictor", page_icon="🏠")  # FIRST command

from app_pages.multipage import MultiPage

# Load page scripts
from app_pages.page_summary import page_summary_body
from app_pages.page_data_study import page_data_study_body
from app_pages.page_house_price_predictor import page_house_price_predictor_body
from app_pages.page_hypothesis_and_validation import page_hypothesis_and_validation_body
from app_pages.page_ML_pipeline import page_ML_pipeline_body

app = MultiPage(app_name="🏡 Heritage House Price Predictor")

# Add pages
app.add_page("Quick Project Summary", page_summary_body)
app.add_page("Data Study & Correlation", page_data_study_body)
app.add_page("House Price Predictor", page_house_price_predictor_body)
app.add_page("Hypotheses and Validation", page_hypothesis_and_validation_body)
app.add_page("ML Pipeline & Performance", page_ML_pipeline_body)

app.run()