import streamlit as st
import pandas as pd
import joblib

# === Cleaned Data ===

@st.cache_data
def load_full_cleaned_data():
    return pd.read_csv("outputs/datasets/cleaned/FullSetCleaned.csv")

@st.cache_data
def load_train_data():
    return pd.read_csv("outputs/datasets/cleaned/TrainSetCleaned.csv")

@st.cache_data
def load_test_data():
    return pd.read_csv("outputs/datasets/cleaned/TestSetCleaned.csv")

# === Prediction Input Files ===

@st.cache_data
def load_inherited_houses():
    return pd.read_csv("outputs/datasets/collection/inherited_houses.csv")

@st.cache_data
def load_price_records():
    return pd.read_csv("outputs/datasets/collection/house_prices_records.csv")

# === Model or Pipeline Loading ===

def load_pkl_file(file_path):
    return joblib.load(filename=file_path)