"""Page 4 — estimate one house with the engineered model.

The same ``engineered_features`` function that built the training data also
transforms the single house you describe here, so the model sees inputs in
exactly the format it learned from.
"""

import pandas as pd
import streamlit as st

from src.featurize import NEIGHBORHOODS
from src.train import predict_price
from src.ui_data import get_model

st.title("Predict Price")

bundle = get_model()

st.markdown("Describe a house and the engineered model estimates its price.")

col1, col2 = st.columns(2)
with col1:
    neighborhood = st.selectbox("Neighborhood", NEIGHBORHOODS)
    sqft = st.slider("Living area (sqft)", 600, 6000, 1700, step=50)
    lot_size = st.slider("Lot size (sqft)", 1000, 60000, 6000, step=500)
    year_built = st.slider("Year built", 1950, 2023, 1995)
with col2:
    bedrooms = st.slider("Bedrooms", 1, 6, 3)
    bathrooms = st.slider("Bathrooms", 1, 4, 2)
    condition = st.slider("Condition (1–5)", 1, 5, 3)
    garage = st.checkbox("Has garage", value=True)

house = pd.DataFrame(
    [
        {
            "neighborhood": neighborhood,
            "sqft": sqft,
            "lot_size": lot_size,
            "year_built": year_built,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "garage": int(garage),
            "condition": condition,
        }
    ]
)

price = predict_price(bundle, house)
st.metric("Estimated price", f"${price:,.0f}")

st.caption(
    "The estimate comes from the engineered linear model. Because the features "
    "are built by the same function used in training, a single house is scored "
    "exactly as the pipeline scored the training set."
)
