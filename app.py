
import pandas as pd
import streamlit as st 
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

from pandas.tseries.offsets import DateOffset
import matplotlib.pyplot as plt

st.title('Model Deployment: SARIMA Model')

st.sidebar.header('User Input Parameters')

def user_input_features():
     Adj Close = st.sidebar.number_input('Insert the value')
     Close = st.sidebar.number_input('Insert the value')
    data = {' Adj Close':Adj Close,
            'Close':Close}
    features = pd.DataFrame(data,index = [0])
    return features 
    
df1 = user_input_features()
st.subheader('User Input parameters')
st.write(df1)


# load the model from disk
df = pd.read_csv("C:\Users\purti\Downloads\AAPL.csv", parse_dates = ['Date'],index_col = 0)

# Split data into train and test
df_train=df.iloc[:1760]   
df_test=df.iloc[1760:]

model2=sm.tsa.statespace.SARIMAX(df_train['Adj Close'], order=(1,1,1),seasonal_order=(1,1,1,63))
model2_fit=model2.fit()


futuredate = [df.index[-1] + DateOffset(days= x) for x in range (0,30)]

Future_dates_df=pd.DataFrame(index=futuredate[1:],columns=df.columns)

future_df=pd.concat([df,Future_dates_df],ignore_index=True)

future_df['forecast']=model2_fit.predict(start=1761,end=(2010 + df1['Close'].iloc[0]-1))


st.subheader('30 days forecast is :')
st.write(future_df['forecast'].tail(30))
