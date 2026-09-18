import streamlit as st
import joblib
import pandas as pd

model = joblib.load("polynomial_regression.pkl")
poly = joblib.load("polynomial_features.pkl")

st.title("Electricity Bill Prediction")

ac_unit = st.number_input(
    "Enter AC Units",
    value=30.0
)

if st.button("Predict"):

    if ac_unit < 1:
        st.error("AC Units cannot be less than 1")

    elif ac_unit > 150:
        st.error("AC Units cannot be greater than 150")

    else:
        new_data = pd.DataFrame({
            "AC_Units": [ac_unit]
        })

        new_data_poly = poly.transform(new_data)

        prediction = model.predict(new_data_poly)

        st.success(f"Predicted Electricity Bill: {prediction[0]:.2f}")
