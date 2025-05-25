import streamlit as st
from app_pages.multipage import MultiPage

# Load page scripts
from app_pages.page_summary import page_summary_body
from app_pages.page_data_study import page_data_study_body
from app_pages.page_house_price_predictor import page_house_price_predictor_body
from app_pages.page_hypothesis_and_validation import page_hypothesis_and_validation_body
from app_pages.page_ML_pipeline import page_ML_pipeline_body

app = MultiPage(app_name="🏡 Heritage House Price Predictor")  # Create app instance

# Add your app pages here using .add_page()
app.add_page("🏠 Quick Project Summary", page_summary_body)
app.add_page("📊 Data Study & Correlation", page_data_study_body)
app.add_page("💰 House Price Predictor", page_house_price_predictor_body)
app.add_page("📑 Hypotheses & Validation", page_hypothesis_and_validation_body)
app.add_page("🧠 ML Pipeline & Performance", page_ML_pipeline_body)

app.run()  # Run the app