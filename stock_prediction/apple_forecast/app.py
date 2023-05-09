import pickle
import streamlit as st
import pandas as pd
import streamlit as st
from plotly import graph_objs as go

st.title('Apple Stock Price Forecast')
st.text('This app forecast the next 30 days Apple stock price by SARIMA model')

with open("sarima_model.pkl", "rb") as f:
    model = pickle.load(f)

value = st.slider("Select no. of days to forecast", 1, 30, 30, step=1)

def main():
    forecast_button = st.button("Forecast")
    if forecast_button:
        forecast = model.get_forecast(steps=value)
        predicted_values = forecast.predicted_mean
        st.subheader("Forecasted Values")
        st.write(predicted_values)
        
        fig = go.Figure()
        df = pd.read_csv("AAPL.csv")
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Actual Stock Price"))
        fig.add_trace(go.Scatter(x=predicted_values.index, y=predicted_values, name="Forecasted Stock Price",line=dict(color='red', width=2)))
        fig.layout.update(title_text='Stock Price forecast with Rangeslider', xaxis_rangeslider_visible=True)
        fig.update_layout(legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
        st.plotly_chart(fig)
if __name__ == "__main__":
    main()
