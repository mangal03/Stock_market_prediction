import pandas as pd
import streamlit as st 
from sklearn.linear_model import LogisticRegression
from pickle import dump
from pickle import load

st.title('Model Deployment: Stock Price Prediction')

st.sidebar.header('User Input Parameters')

def user_input_features():
    Close = st.sidebar.number_input("Insert the Number")
    Adj Close = st.sidebar.number_input("Insert the Number")
    data = {'Close': Close,
            'Adj Close': Adj Close}
    features = pd.DataFrame(data,index = [0])
    return features 
    
df = user_input_features()
st.subheader('User Input parameters')
st.write(df)


# load the model from disk
loaded_model = load(open('ARIMA_Model.sav', 'rb'))

prediction = loaded_model.predict(data)
prediction_proba = loaded_model.predict_proba(data)

st.subheader('Predicted Result')
st.write('Yes' if prediction_proba[0][1] > 0.5 else 'No')

st.subheader('Prediction Probability')
st.write(prediction_proba)

