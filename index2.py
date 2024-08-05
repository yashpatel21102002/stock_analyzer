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
NEWS_API_KEY = 'b54e87e131fc49ddba85749699aa962b'
NEWS_API_URL = 'https://newsapi.org/v2/everything'
#

# App configuration
st.set_page_config(page_title="Stock Analyzer", layout="wide")

st.write("<p style='color: red;'>* Note: Forecasting is based on historical data and may highly misleading. It is only fun project.</p>", unsafe_allow_html=True)


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
                'Price': stock_data.get('regularMarketPrice', np.float64('nan')),
                'P/E Ratio': stock_data.get('trailingPE', np.float64('nan')),
                'EPS': stock_data.get('earningsPerShare', np.float64('nan')),
                'Market Cap': stock_data.get('marketCap', np.float64('nan')),
                'Dividend Yield': stock_data.get('dividendYield', np.float64('nan')),
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
            <p style='font-size: 16px; margin: 0;'><strong>Day Gain:</strong> <span style='color: {"#27ae60" if isinstance(day_gain, (int, np.float64)) and day_gain >= 0 else "#e74c3c"};'>{day_gain if day_gain != 'N/A' else 'N/A'} ({day_gain_percent if day_gain_percent != 'N/A' else 'N/A'}%)</span></p>
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
st.header("")
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

# Add PE and EPS if available
if 'trailingPE' in stock_info:
    fig.add_trace(
        go.Scatter(x=data['Date'], y=np.repeat(stock_info['trailingPE'], len(data)), mode='lines', name='PE Ratio', line=dict(color='green')),
        row=1, col=1
    )
if 'earningsPerShare' in stock_info:
    fig.add_trace(
        go.Scatter(x=data['Date'], y=np.repeat(stock_info['earningsPerShare'], len(data)), mode='lines', name='EPS', line=dict(color='purple')),
        row=1, col=1
    )


fig.update_layout(title=f'{selected_stock} Historical Data', xaxis_title='Date', yaxis_title='Price')
st.plotly_chart(fig, use_container_width=True)

# Forecasting
# Forecasting
st.subheader('🔮 Forecasting')

# Years of prediction slider
n_years = st.slider("Years of Prediction:", 1, 5)
period = n_years * 365

# Preparing data for forecasting
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

# Creating and fitting the model
m = Prophet()
m.add_country_holidays(country_name='IN')
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Plot forecast
fig1 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.2)

# Main Forecast Line
fig1.add_trace(go.Scatter(
    x=forecast['ds'], y=forecast['yhat'], name='Forecast', line=dict(color='royalblue', width=2)
), row=1, col=1)

# Confidence Intervals
fig1.add_trace(go.Scatter(
    x=forecast['ds'], y=forecast['yhat_upper'], name='Upper Bound', line=dict(color='lightblue', width=1, dash='dash')
), row=1, col=1)

fig1.add_trace(go.Scatter(
    x=forecast['ds'], y=forecast['yhat_lower'], name='Lower Bound', line=dict(color='lightblue', width=1, dash='dash')
), row=1, col=1)

# Trend and Seasonality Components
fig1.add_trace(go.Scatter(
    x=forecast['ds'], y=forecast['trend'], name='Trend', line=dict(color='red', width=2)
), row=2, col=1)

fig1.update_layout(
    title="Forecast with Components",
    xaxis_title="Date",
    yaxis_title="Price",
    height=1000,
    width=1200,
    template='plotly_dark'
)

# Display forecast graph
st.plotly_chart(fig1)

#-------------------------------trend analysis -------------------
st.subheader('📈 Trend Analysis and Buy Recommendation')

# Calculate short-term (30 days) and long-term (365 days) returns
data['Return'] = data['Close'].pct_change() * 100
short_term_return = data['Return'].tail(30).mean()*100
long_term_return = data['Return'].tail(365).mean()*100

current_trend = "Bullish" if short_term_return > 0 else "Bearish"
recommendation = "Buy" if short_term_return > 0 else "Sell"

st.markdown(
    f"""
    <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; background-color: #2c3e50; color: #ecf0f1; padding: 20px; border-radius: 12px; box-shadow: 0 6px 12px rgba(0,0,0,0.4); font-family: Arial, sans-serif;'>
        <div style='background-color: #34495e; padding: 15px; border-radius: 8px;'>
            <p style='font-size: 16px; margin: 0;'><strong>Current Trend:</strong> <span style='color: {"#27ae60" if current_trend == "Bullish" else "#e74c3c"};'>{current_trend}</span></p>
        </div>
        <div style='background-color: #34495e; padding: 15px; border-radius: 8px;'>
            <p style='font-size: 16px; margin: 0;'><strong>Short-term Return (30 days):</strong> <span style='color: {"#27ae60" if short_term_return > 0 else "#e74c3c"};'>{short_term_return:.2f}%</span></p>
        </div>
        <div style='background-color: #34495e; padding: 15px; border-radius: 8px;'>
            <p style='font-size: 16px; margin: 0;'><strong>Long-term Return (365 days):</strong> <span style='color: {"#27ae60" if long_term_return > 0 else "#e74c3c"};'>{long_term_return:.2f}%</span></p>
        </div>
        <div style='background-color: #34495e; padding: 15px; border-radius: 8px;'>
            <p style='font-size: 16px; margin: 0;'><strong>Buy Recommendation:</strong> <span style='color: {"#27ae60" if recommendation == "Buy" else "#e74c3c"};'>{recommendation}</span></p>
        </div>
    </div>
    """, unsafe_allow_html=True
)
#----------------------------------------------------------



# Normalize values
sector_info['Normalized Market Cap'] = (sector_info['Market Cap'] - sector_info['Market Cap'].min()) / (sector_info['Market Cap'].max() - sector_info['Market Cap'].min())
sector_info['Normalized P/E Ratio'] = (sector_info['P/E Ratio'] - sector_info['P/E Ratio'].min()) / (sector_info['P/E Ratio'].max() - sector_info['P/E Ratio'].min())
sector_info['Normalized Dividend Yield'] = (sector_info['Dividend Yield'] - sector_info['Dividend Yield'].min()) / (sector_info['Dividend Yield'].max() - sector_info['Dividend Yield'].min())

# Calculate average P/E Ratio for the horizontal line
average_pe = sector_info['P/E Ratio'].mean()

# Plotting
st.subheader('📊 Sector Analysis')
if not sector_info.empty:
    fig3 = make_subplots(rows=1, cols=1)

    # Bar for Market Cap
    fig3.add_trace(
        go.Bar(
            x=sector_info['Ticker'],
            y=sector_info['Normalized Market Cap'],
            name='Market Cap',
            marker_color='royalblue',
            hovertext=sector_info['Market Cap'],
            hoverinfo='x+text'
        ),
        row=1, col=1
    )

    # Bar for P/E Ratio
    fig3.add_trace(
        go.Bar(
            x=sector_info['Ticker'],
            y=sector_info['Normalized P/E Ratio'],
            name='P/E Ratio',
            marker_color='orange',
            hovertext=sector_info['P/E Ratio'],
            hoverinfo='x+text'
        ),
        row=1, col=1
    )

    # Bar for Dividend Yield
    fig3.add_trace(
        go.Bar(
            x=sector_info['Ticker'],
            y=sector_info['Normalized Dividend Yield'],
            name='Dividend Yield',
            marker_color='green',
            hovertext=sector_info['Dividend Yield'],
            hoverinfo='x+text'
        ),
        row=1, col=1
    )

    # Add horizontal line for average P/E Ratio
    fig3.add_hline(y=(average_pe - sector_info['P/E Ratio'].min()) / (sector_info['P/E Ratio'].max() - sector_info['P/E Ratio'].min()), line_dash="dash", line_color="red", annotation_text="Average P/E Ratio", annotation_position="top left")

    fig3.update_layout(
        title=f'{selected_stock} Sector Analysis',
        xaxis_title='Company',
        yaxis_title='Normalized Value',
        barmode='group'
    )
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.write("No sector data available for this stock.")


#----------------------------------------------------------------------
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
# Educational Information
educational_text = """
### How This App Works

1. **Libraries Used**: The app uses the following libraries:
    - `pandas` for data manipulation.
    - `numpy` for numerical operations.
    - `yfinance` to fetch the financial and stock data.
    - `Prophet` for predicting the future stock prices.
    - `plotly` for interactive plotting.

2. **Stock Data Fetching**: The stock data is fetched using the `yfinance` library, which provides historical stock prices and other financial data.

3. **Prediction Algorithm**:
    - The app uses the `Prophet` library developed by Facebook for forecasting. Prophet is particularly good at capturing seasonality effects and trend changes in the data.
    - **Model Components**: Prophet decomposes the time series data into three main components:
        - **Trend**: The non-periodic changes in the value.
        - **Seasonality**: The periodic changes that occur at fixed periods (e.g., yearly, weekly).
        - **Holidays**: The effects of holidays which are specified by the user.
    - **Additive Model**: Prophet uses an additive model where the predicted value is the sum of these components.
    - **Algorithm**: The underlying algorithm involves fitting piecewise linear or logistic growth curves to the trend component, which allows it to adapt to changes in trend direction.
    - **Seasonality**: Prophet handles seasonality by using Fourier series to provide a flexible yet interpretable model for periodic changes.
    - **Holidays and Events**: Users can specify holidays and special events, which are included in the model to account for their impact on the time series.

4. **Visualization**:
    - The historical and predicted stock prices are visualized using `plotly` to provide an interactive and informative experience.
"""

# Display educational information in red text
st.markdown(f"<div style='color: red;'>{educational_text}</div>", unsafe_allow_html=True)

# Footer
footer_text = """
<hr style="border:1px solid gray"> </hr>
<div style='text-align: center; padding: 10px;'>
    <p style='color: gray;'>Made with 💘 by Yash Patel</p>
    <p style='color: gray;'>© 2024 Syz Technologies</p>
    <p style='color: gray;'>For more information, visit our <a href='https://www.syztechnologies.com' target='_blank' style='color: gray;'>website</a>.</p>
</div>
"""

st.markdown(footer_text, unsafe_allow_html=True)



