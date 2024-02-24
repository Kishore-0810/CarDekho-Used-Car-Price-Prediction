# importing the necessary libraries
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import pickle


# reading cleaned data file
df = pd.read_csv("Cleaned_car_dataset.csv")


# # function to predict car price
def Price_prediction(Engine_Displacement, Model_Manufactured_Year, Body_Type, Transmission, Model, Mileage, Engine_Type, Location, Km_Driven):

    Body_Type = df["bt_encode"][df["bt"] == f"{Body_Type}"].unique()[0]
    Location = df["Location_encode"][df["Location"] == f"{Location}"].unique()[0]
    Transmission = df["Transmission_encode"][df["Transmission"] == f"{Transmission}"].unique()[0]
    Model = df["model_encode"][df["model"] == f"{Model}"].unique()[0]
    Engine_Type = df["Engine Type_encode"][df["Engine Type"] == f"{Engine_Type}"].unique()[0]

    with open("used_car_prediction_model.pkl", "rb") as file:
        model = pickle.load(file)

    prediction = model.predict([[Engine_Displacement, Model_Manufactured_Year, Body_Type, Transmission, Model, Mileage, Engine_Type, Location, Km_Driven]])

    return np.exp(prediction)



# streamlit setup
st.set_page_config("Car Dekho Used Car Price Prediction", layout = "wide")


selected = option_menu(None, 
                       options = ["Menu", "Prediction"],
                       icons = ["house"],
                       orientation = "horizontal",
                       styles = {"nav-link": {"font-size": "18px", "text-align": "center", "margin": "1px"},
                                 "icon": {"color": "yellow", "font-size": "20px"},
                                 "nav-link-selected": {"background-color": "#9457eb"}})


if selected == "Menu":
    
    st.title(":red[**Car Dekho Used Car Price Prediction**]")

    st.markdown("")

    st.markdown('''* Given a diverse dataset containing information about used cars, including features like car model, number of owners, age, 
                     mileage, fuel type, kilometers driven, features, and location, the objective is to build an accurate machine learning model.''') 
                
    st.markdown('''* This model will predict the current valuations for used cars, enabling users to make informed decisions when buying or 
                     selling pre-owned vehicles.''')


if selected == "Prediction":

    with st.form("Car Price Prediction"):

        col1, col2 = st.columns(2)

        with col1:

            st.selectbox(":blue[**Model**]", options = df["model"].unique(), key = "m")

            st.number_input(":blue[**Model Manufactured Year**]", min_value = 1900 , step = 1, key = "mmy")

            st.selectbox(":blue[**Engine Type**]", options = df["Engine Type"].unique(), key = "et")

            st.number_input(":blue[**Engine Displacement**]", min_value = 0, key = "ed")

        with col2:

            st.number_input(":blue[**Km Driven**]", min_value = 0, key = "kd")

            st.number_input(":blue[**Mileage**]", min_value = 0.0, key = "ml")

            st.selectbox(":blue[**Body Type**]", options = df["bt"].unique(), key = "bt")

            st.radio(":blue[**Transmission**]", options = ["Manual", "Automatic"], key = "tm" )
            
            st.selectbox(":blue[**Location**]", options = df["Location"].unique(), key = "l")

            
        with col1:

            if st.form_submit_button("**Predict**"):
                
                pred = Price_prediction(st.session_state["ed"], st.session_state["mmy"], st.session_state["bt"], 
                                        st.session_state["tm"], st.session_state["m"], st.session_state["ml"], 
                                        st.session_state["et"], st.session_state["l"], st.session_state["kd"])
                
                st.success(f"The Predicted Price is :green[₹ {pred[0]:,.2f} Lakhs]")



# --------------------------x---------------------------------x-------------------------------------x------------------------------------x---------------------------------