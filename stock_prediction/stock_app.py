import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from plotly import graph_objs as go
import streamlit as st

st.title('Stock Price Prediction')
st.text('This app predicts the next 15 days stock price by using LSTM Machine learning model \nand the results take 30sec to load')
user_input = st.text_input('Enter Stock Ticker','AAPL')

df = yf.download(tickers=user_input, period='3y')
y = df['Close'].fillna(method='ffill')
y = y.values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(y)
y = scaler.transform(y)

n_lookback = 60 
n_prediction = 15

X = []
Y = []

for i in range(n_lookback, len(y) - n_prediction + 1):
    X.append(y[i - n_lookback: i])
    Y.append(y[i: i + n_prediction])

X = np.array(X)
Y = np.array(Y)

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(n_lookback, 1)))
model.add(LSTM(units=50))
model.add(Dense(n_prediction))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X, Y, epochs=10, batch_size=32, verbose=0)

X_pred = y[- n_lookback:]
X_pred = X_pred.reshape(1, n_lookback, 1)
Y_pred = model.predict(X_pred).reshape(-1, 1)
Y_pred = scaler.inverse_transform(Y_pred)

# organize the results in a data frame
df_past = df[['Close']].reset_index()
df_past.rename(columns={'index': 'Date', 'Close': 'Actual'}, inplace=True)
df_past['Date'] = pd.to_datetime(df_past['Date'])
df_past['Prediction'] = np.nan
df_past['Prediction'].iloc[-1] = df_past['Actual'].iloc[-1]

df_future = pd.DataFrame(columns=['Date', 'Actual', 'Prediction'])
df_future['Date'] = pd.date_range(start=df_past['Date'].iloc[-1] + pd.Timedelta(days=1), periods=n_prediction)
df_future['Prediction'] = Y_pred.flatten()
df_future['Actual'] = np.nan

results = df_past.append(df_future).set_index('Date')

fig = go.Figure()
fig.add_trace(go.Scatter(x=results.index, y=results['Actual'], name="Actual Stock Price"))
fig.add_trace(go.Scatter(x=results.index, y=results['Prediction'], name="Predicted Stock Price",line=dict(color='red', width=4)))
fig.layout.update(title_text='15 days Stock Price Prediction with Rangeslider', xaxis_rangeslider_visible=True)
fig.update_layout(legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
st.plotly_chart(fig)

ma10 = df.Close.rolling(10).mean()
ma50 = df.Close.rolling(50).mean()
ma100 = df.Close.rolling(100).mean()
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Actual Stock Price"))
fig.add_trace(go.Scatter(x=ma10.index, y=ma10, name="10 Moving Avg",line=dict(color='red')))
fig.add_trace(go.Scatter(x=ma50.index, y=ma50, name="50 Moving Avg",line=dict(color='yellow')))
fig.add_trace(go.Scatter(x=ma100.index, y=ma100, name="100 Moving Avg",line=dict(color='green')))
fig.layout.update(title_text='Moving Average Stock Price data', xaxis_rangeslider_visible=True)
fig.update_layout(legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
st.plotly_chart(fig)

st.subheader('Last 1 week stock price')
st.write(df.tail(7))

st.subheader("About")
st.text("Built by Srikar Thadaka")
    
@st.cache(allow_output_mutation=True)
def Pageviews():
    return []

pageviews=Pageviews()
pageviews.append('dummy')

try:
    st.markdown('v. {}'.format(len(pageviews)))
except ValueError:
    st.markdown('v. {}'.format(1))