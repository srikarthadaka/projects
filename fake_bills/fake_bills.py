import streamlit as st
import pandas as pd
import pickle as pkl
import xgboost as xgb

model = xgb.XGBClassifier()
model.load_model('finalized_model_xgb.sav')

# @st.cache
# model = pkl.load(open('finalized_model_xgb.sav','rb'))
# data = pd.read_csv("https://raw.githubusercontent.com/srikarthadaka/projects/main/fake_bills/fake_bills.csv", sep=';')

def predict(diagonal, height_left, height_right, margin_low, margin_up, length):
    prediction = model.predict(pd.DataFrame([[diagonal, height_left, height_right, margin_low, margin_up, length]], 
                                            columns=['diagonal', 'height_left', 'height_right', 'margin_low', 'margin_up', 'length']))
    return prediction

def main():
    menu = ["Home","About"]
    choice = st.sidebar.selectbox("Menu",menu)
    
    if choice == "Home":
        st.title("Fake Bills Classification")
        st.markdown(""" This App is going to classify reciet bills are Fake or Genuine. This Classification is done by
                    XGBoost and Random Forest Machine Learning methods.""")
        st.text("Sample data") 
        data = pd.read_csv("https://raw.githubusercontent.com/srikarthadaka/projects/main/fake_bills/fake_bills.csv", sep=';')
        st.dataframe(data.iloc[997:1003])
        
        st.sidebar.header('User Input Dimensions')
        diagonal = st.sidebar.slider('Diagonal:', min_value=170.00, max_value=174.00, value=171.00)
        height_left = st.sidebar.slider('Height Left:', min_value=100.00, max_value=105.00, value=100.00)
        height_right = st.sidebar.slider('Height Right:', min_value=100.00, max_value=105.00, value=100.00)
        margin_low = st.sidebar.slider('Margin Low:', min_value=2.00, max_value=5.00, value=2.00)
        margin_up = st.sidebar.slider('Margin Up:', min_value=2.00, max_value=5.00, value=2.00)
        length = st.sidebar.slider('Length:', min_value=109.00, max_value=115.00, value=109.00)
 
        st.text("Please input bills dimensions to the left of this webpage.") 
        if st.button('Predict'):
            result = predict(diagonal, height_left, height_right, margin_low, margin_up, length)
            if result == 0:
                st.success("The reciet bill is:   Fake")
            elif result == 1:
                st.success("The reciet bill is:   Genuine")
            else:
                st.success("Invalid value")
    
    else:
        st.subheader("About")
        st.success("Built by Srikar Thadaka")

if __name__ == '__main__':
    main()
    
