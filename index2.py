import streamlit as st
import datetime as dt
import yfinance as yf
from prophet import Prophet
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import requests
import pandas as pd
import numpy as np

# Constants
START = "2015-01-01"
TODAY = dt.date.today().strftime("%Y-%m-%d")
NEWS_API_KEY = 'YOUR_NEWSAPI_KEY'
NEWS_API_URL = 'https://newsapi.org/v2/everything'

# App configuration
st.set_page_config(page_title="Stock Analyzer", layout="wide")
st.title("📈 Stock Analyzer")

# Function to load data
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

# Function to get latest data
@st.cache_data
def get_latest_data(ticker):
    ticker_data = yf.Ticker(ticker)
    todays_data = ticker_data.history(period='1d')
    return todays_data

# Function to get stock info
@st.cache_data
def get_stock_info(ticker):
    ticker_data = yf.Ticker(ticker)
    return ticker_data.info

# Function to get news articles
@st.cache_data
def get_news(ticker):
    company_name = yf.Ticker(ticker).info['shortName']
    response = requests.get(NEWS_API_URL, params={
        'q': company_name,
        'apiKey': NEWS_API_KEY,
        'language': 'en',
        'sortBy': 'relevancy'
    })
    articles = response.json().get('articles', [])
    return articles

# Function to get sector information
@st.cache_data
def get_sector_info(stock_ticker):
    sector_stocks = {
        "Technology": ["TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS", "LTI.NS"],
        "Finance": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS", "HDFC.NS"],
        "Healthcare": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "LUPIN.NS", "APOLLOHOSP.NS", "DIVISLAB.NS"],
        "Energy": ["RELIANCE.NS", "ONGC.NS", "ADANIGREEN.NS", "NTPC.NS", "POWERGRID.NS", "GAIL.NS"],
        "Consumer": ["HINDUNILVR.NS", "ITC.NS", "DABUR.NS", "MARICO.NS", "TATAMOTORS.NS", "MCDOWELL-N.NS"],
        "Automobile": ["TATAMOTORS.NS", "MARUTI.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS", "EICHERMOT.NS"],
        "Utilities": ["NTPC.NS", "POWERGRID.NS", "GAIL.NS", "ADANIGREEN.NS", "TATAELXSI.NS"]
    }
    
    sector = None
    for key, stocks in sector_stocks.items():
        if stock_ticker in stocks:
            sector = key
            break

    if not sector:
        return pd.DataFrame()  # Return empty dataframe if sector is not found

    stocks = sector_stocks[sector]
    sector_data = []
    for stock in stocks:
        try:
            stock_data = yf.Ticker(stock).info
            sector_data.append({
                'Ticker': stock,
                'Company': stock_data.get('shortName', stock),
                'Price': stock_data.get('regularMarketPrice', float('nan')),
                'P/E Ratio': stock_data.get('trailingPE', float('nan')),
                'EPS': stock_data.get('earningsPerShare', float('nan')),
                'Market Cap': stock_data.get('marketCap', float('nan')),
                'Dividend Yield': stock_data.get('dividendYield', float('nan')),
            })
        except:
            continue
    return pd.DataFrame(sector_data)

# Sidebar for stock selection
st.sidebar.header("Stock Selection")
stocks = (
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
    "ICICIBANK.NS", "KOTAKBANK.NS", "SBIN.NS", "BAJFINANCE.NS", "BHARTIARTL.NS",
    "ASIANPAINT.NS", "HCLTECH.NS", "ITC.NS", "LT.NS", "AXISBANK.NS",
    "WIPRO.NS", "HDFCLIFE.NS", "ULTRACEMCO.NS", "TECHM.NS", "SUNPHARMA.NS",
    "TITAN.NS", "MARUTI.NS", "ADANIPORTS.NS", "POWERGRID.NS", "NTPC.NS",
    "GRASIM.NS", "JSWSTEEL.NS", "NESTLEIND.NS", "DRREDDY.NS", "INDUSINDBK.NS",
    "TATAMOTORS.NS", "TATAPOWER.NS", "NHPC.NS", "ADANIGREEN.NS", "IDEA.NS"
)
selected_stock = st.sidebar.selectbox("Select Stock for Analysis", stocks)
forecast_period = st.sidebar.slider('Forecast Period (Days)', min_value=30, max_value=1825, value=365, step=30)

# Fetch data
data_load_state = st.text("Fetching data... please wait!")
data = load_data(selected_stock)
latest_data = get_latest_data(selected_stock)
stock_info = get_stock_info(selected_stock)
news_articles = get_news(selected_stock)
sector_info = get_sector_info(selected_stock)
data_load_state.text("Data is in! Time to analyze.")

# Stock metrics
current_price = latest_data['Close'][0]
previous_close = latest_data['Open'][0]
day_gain = current_price - previous_close
day_gain_percent = (day_gain / previous_close) * 100
volume = latest_data['Volume'][0]

# Display stock information
st.subheader('📊 Stock Overview')
st.markdown(
    f"""
    <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; background-color: #2c3e50; color: #ecf0f1; padding: 20px; border-radius: 12px; box-shadow: 0 6px 12px rgba(0,0,0,0.4); font-family: Arial, sans-serif;'>
        <div style='background-color: #34495e; padding: 15px; border-radius: 8px;'>
            <p style='font-size: 16px; margin: 0;'><strong>Current Price:</strong> <span style='color: #1abc9c;'>{current_price if current_price != 'N/A' else 'N/A'} INR</span></p>
        </div>
        <div style='background-color: #34495e; padding: 15px; border-radius: 8px;'>
            <p style='font-size: 16px; margin: 0;'><strong>Day Gain:</strong> <span style='color: {"#27ae60" if isinstance(day_gain, (int, float)) and day_gain >= 0 else "#e74c3c"};'>{day_gain if day_gain != 'N/A' else 'N/A'} ({day_gain_percent if day_gain_percent != 'N/A' else 'N/A'}%)</span></p>
        </div>
        <div style='background-color: #34495e; padding: 15px; border-radius: 8px;'>
            <p style='font-size: 16px; margin: 0;'><strong>Volume:</strong> {volume if volume != 'N/A' else 'N/A'}</p>
        </div>
        <div style='background-color: #34495e; padding: 15px; border-radius: 8px;'>
            <p style='font-size: 16px; margin: 0;'><strong>Previous Close:</strong> {previous_close if previous_close != 'N/A' else 'N/A'}</p>
        </div>
        <div style='background-color: #34495e; padding: 15px; border-radius: 8px;'>
            <p style='font-size: 16px; margin: 0;'><strong>Market Cap:</strong> {stock_info.get('marketCap', 'N/A')}</p>
        </div>
        <div style='background-color: #34495e; padding: 15px; border-radius: 8px;'>
            <p style='font-size: 16px; margin: 0;'><strong>P/E Ratio:</strong> {stock_info.get('trailingPE', 'N/A')}</p>
        </div>
        <div style='background-color: #34495e; padding: 15px; border-radius: 8px;'>
            <p style='font-size: 16px; margin: 0;'><strong>EPS:</strong> {stock_info.get('earningsPerShare', 'N/A')}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Historical chart
st.subheader('📈 Historical Data')
fig = make_subplots(rows=1, cols=1)
fig.add_trace(
    go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Closing Price', line=dict(color='blue')),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=data['Date'], y=data['Close'].rolling(window=100).mean(), mode='lines', name='100-Day MA', line=dict(color='orange')),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=data['Date'], y=data['Close'].rolling(window=200).mean(), mode='lines', name='200-Day MA', line=dict(color='red')),
    row=1, col=1
)
fig.update_layout(title=f'{selected_stock} Historical Data', xaxis_title='Date', yaxis_title='Price')
st.plotly_chart(fig, use_container_width=True)

# Forecasting
st.subheader('🔮 Forecasting')
data_prophet = data.rename(columns={'Date': 'ds', 'Close': 'y'})
model = Prophet()
model.fit(data_prophet)
future = model.make_future_dataframe(periods=forecast_period)
forecast = model.predict(future)
fig2 = model.plot(forecast)
st.plotly_chart(fig2, use_container_width=True)

# Sector analysis
st.subheader('📊 Sector Analysis')
if not sector_info.empty:
    fig3 = make_subplots(rows=1, cols=1)
    fig3.add_trace(
        go.Bar(x=sector_info['Ticker'], y=sector_info['Market Cap'], name='Market Cap', marker_color='royalblue'),
        row=1, col=1
    )
    fig3.add_trace(
        go.Bar(x=sector_info['Ticker'], y=sector_info['P/E Ratio'], name='P/E Ratio', marker_color='orange'),
        row=1, col=1
    )
    fig3.add_trace(
        go.Bar(x=sector_info['Ticker'], y=sector_info['EPS'], name='EPS', marker_color='green'),
        row=1, col=1
    )
    fig3.update_layout(title=f'{selected_stock} Sector Analysis', xaxis_title='Company', yaxis_title='Value')
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.write("No sector data available for this stock.")

# News section
st.subheader('📰 Latest News')
if news_articles:
    for article in news_articles:
        st.write(f"**{article['title']}**")
        st.write(article['description'])
        st.write(f"[Read more]({article['url']})")
        st.write("----")
else:
    st.write("No news articles found.")

# Notes
st.write("<p style='color: red;'>* Note: Forecasting is based on historical data and may not always be accurate. Use caution while making investment decisions.</p>", unsafe_allow_html=True)

