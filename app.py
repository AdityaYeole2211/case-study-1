import streamlit as st
import numpy as np
import pickle



try:
    with open('xgb_model.pkl', 'rb') as file:
        model = pickle.load(file)
    model_loaded = True
except Exception as e:
    st.error("🚫 Failed to load the prediction model.")
    st.error(f"Details: {e}")
    model_loaded = False


st.title("🏡 House Price Prediction App")


bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, step=1)
bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10, step=1)
size = st.number_input("Size (in sqft)", min_value=100.0, max_value=10000.0, step=50.0)
year_built = st.number_input("Year Built", min_value=1900, max_value=2025, step=1)


sold_date = st.date_input("Date Sold")
sold_year = sold_date.year
sold_month = sold_date.month

condition = st.selectbox("Condition", ['Poor', 'Fair', 'Good', 'New'])
location = st.selectbox("Location", ['CityA', 'CityB', 'CityC', 'CityD'])
type_ = st.selectbox("Property Type", ['Single Family', 'Townhouse', 'Condominium'])


# Total rooms
total_rooms = bedrooms + bathrooms


condition_options = ['Fair', 'Good', 'New', 'Poor']
condition_encoded = [1 if condition == cond else 0 for cond in condition_options]

location_options = ['CityA', 'CityB', 'CityC', 'CityD']
location_encoded = [1 if location == loc else 0 for loc in location_options]

type_options = ['Condominium', 'Single Family', 'Townhouse']
type_encoded = [1 if type_ == t else 0 for t in type_options]


features = np.array([[ 
    bedrooms,
    bathrooms,
    size,
    year_built,
    sold_year,
    sold_month,
    total_rooms,
    *condition_encoded,
    *location_encoded,
    *type_encoded
]])


if st.button("Predict House Price"):
    log_price = model.predict(features)[0]
    predicted_price = np.expm1(log_price)  # reverse log(1 + price)
    st.success(f"Estimated House Price: ${predicted_price:,.2f}")
