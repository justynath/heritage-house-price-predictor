import pandas as pd
import numpy as np

def predict_sale_price(X_live, pipeline):
    """
    Accepts user input features and a trained pipeline.
    Returns a predicted sale price.
    """
    prediction = pipeline.predict(X_live)
    return np.round(prediction[0], 2)