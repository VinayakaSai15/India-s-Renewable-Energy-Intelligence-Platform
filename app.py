# import streamlit as st
# from models_6_to_9 import *

import streamlit as st
from models_6_to_9 import get_2024_data


st.title("🌱 Renewable Energy Platform")

if st.button("Load 2024 Data"):
    df = get_2024_data()
    st.write(df)

st.title("🌱 Renewable Energy Intelligence Platform")

st.write("Welcome to your dashboard!")

# Example button to trigger your model
if st.button("Run Models"):
    st.write("Running predictions...")

    # call your functions here
    # example:
    # result = your_function()
    # st.write(result)