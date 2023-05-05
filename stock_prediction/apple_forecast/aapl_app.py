import pandas as pd
import streamlit as st
from plotly import graph_objs as go
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.title('Apple Stock Price Forecast')
st.text('This app forecast the next 30 days Apple stock price by SARIMA model')

df = pd.read_csv("https://raw.githubusercontent.com/srikarthadaka/projects/main/stock_prediction/apple_forecast/aapl_data.csv",
                 index_col='Date')

value = st.slider("Select a value", 1, 30, step=1)

def forecast(value):
    model_sarima = SARIMAX(df["Close"], order=(2,1,2), seasonal_order=(2,1,2,5))
    model_sarima_fit = model_sarima.fit()
    last_date = df.index[-1]
    date_range = pd.date_range(last_date, periods=value+1, freq='D')[1:]
    results = model_sarima_fit.forecast(steps=value)
    results.index = date_range
    return results

if st.button('Generate Forecast'):
    results = forecast(value)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Actual Stock Price"))
    fig.add_trace(go.Scatter(x=results.index, y=results, name="Forecasted Stock Price",line=dict(color='red', width=2)))
    fig.layout.update(title_text='Stock Price forecast with Rangeslider', xaxis_rangeslider_visible=True)
    fig.update_layout(legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
    st.plotly_chart(fig)

    results_df = pd.DataFrame(results)
    st.table(results_df)
