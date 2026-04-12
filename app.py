import streamlit as st
from models_6_to_9 import *

df = load_data()
st.write(df.head())

st.title("🌱 Renewable Energy Intelligence Platform")

st.write("Welcome to your dashboard!")

# Example button to trigger your model
if st.button("Run Models"):
    st.write("Running predictions...")

    # call your functions here
    # example:
    # result = your_function()
    # st.write(result)