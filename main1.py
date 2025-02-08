import os
import requests
import csv
from fastapi import FastAPI
from pydantic import BaseModel
from textblob import TextBlob
import tweepy
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from fastapi.responses import JSONResponse
from statsmodels.tsa.arima.model import ARIMA
from fastapi import FastAPI, HTTPException
import numpy as np
import pandas as pd

app = FastAPI()

# Set up Twitter API (replace with your own API credentials)
BEARER_TOKEN = "your_twitter_bearer_token_here"
client = tweepy.Client(bearer_token=BEARER_TOKEN)

# Directory to store stock data CSV files
DATA_DIR = "data/stock_data/"
os.makedirs(DATA_DIR, exist_ok=True)

# Pydantic model to get stock symbol from user
class StockRequest(BaseModel):
    stock_symbol: str


# Function to fetch and preprocess stock data
# Function to fetch and preprocess stock data
def get_stock_data(stock_symbol: str):
    # Download stock data from Yahoo Finance
    stock_data = yf.download(stock_symbol, period="2y", interval="1d")
    
    # Check if the data is empty
    if stock_data.empty:
        raise ValueError(f"No data found for symbol {stock_symbol}")
    
    # Use the 'Close' price and drop rows with NaN values
    stock_data = stock_data[['Close']].dropna()
    
    # Normalize data using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_data[['Close']].values.reshape(-1, 1))
    
    return stock_data, scaled_data, scaler


# Function to preprocess data for LSTM
def preprocess_data(stock_data, scaled_data, time_steps=60):
    x_data = []
    y_data = []
    
    for i in range(time_steps, len(scaled_data)):
        x_data.append(scaled_data[i-time_steps:i, 0])  # Create the input sequence of size 'time_steps'
        y_data.append(scaled_data[i, 0])  # The target is the next day's price
    
    x_data = np.array(x_data)  # Convert to numpy array for LSTM
    y_data = np.array(y_data)
    
    # Reshape data for LSTM (samples, time_steps, features)
    x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], 1))
    
    return x_data, y_data

# Function to fetch Twitter data based on stock symbol
def get_twitter_data(stock_symbol: str):
    query = f"{stock_symbol} -is:retweet lang:en"
    tweets = client.search_recent_tweets(query=query, tweet_fields=['created_at', 'public_metrics'], max_results=100)
    return tweets.data if tweets.data else []

# Function for sentiment analysis using TextBlob for Twitter data
def analyze_sentiment_twitter(tweets):
    sentiment_scores = []
    for tweet in tweets:
        analysis = TextBlob(tweet.text)
        sentiment_scores.append(analysis.sentiment.polarity)
    return sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0

# Function to fetch Alpha Vantage news and analyze sentiment
def get_alpha_vantage_news(stock_symbol):
    API_KEY = 'N93WB061BPO3QZ7M'
    url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={stock_symbol}&apikey={API_KEY}'
    response = requests.get(url)
    data = response.json()
    news = [{'headline': article['title'], 'link': article['url']} for article in data.get('feed', [])]
    return news

def analyze_sentiment_news(news_data):
    sentiment_scores = []
    for article in news_data:
        analysis = TextBlob(article['headline'])
        sentiment_scores.append(analysis.sentiment.polarity)
    return sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0

# Function to predict stock price using ARIMA model
def predict_stock_arima(stock_data):
    try:
        # Ensure the data is univariate (use only one column like 'Close' or 'Adj Close')
        stock_data = stock_data['Close']
        stock_data = stock_data.dropna()  # Remove NaN values

        # Fit ARIMA model
        model = ARIMA(stock_data, order=(1, 1, 1))
        model_fit = model.fit()

        # Make prediction
        forecast = model_fit.forecast(steps=1)  # Forecast the next point
        predicted_price = forecast[0]  # Get the predicted value

        return {"predicted_price": predicted_price}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")
    

# Function to preprocess stock data for Linear Regression
def preprocess_stock_data(stock_data):
    stock_data['Prev Close'] = stock_data['Close'].shift(1)
    stock_data = stock_data.dropna()
    
    # Features and target
    X = stock_data[['Prev Close']]  # Use previous day's close price as feature
    y = stock_data['Close']  # Predict the current day's close price
    
    return X, y

# Train a Linear Regression model
def train_linear_regression_model(stock_symbol):
    stock_data = get_stock_data(stock_symbol)
    X, y = preprocess_stock_data(stock_data)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate the model
    score = model.score(X_test, y_test)
    print(f"Model R^2 Score: {score}")
    
    return model

# Function to predict the next stock price using Linear Regression model
def predict_stock_linear_regression(stock_symbol):
    stock_data = get_stock_data(stock_symbol)
    X, _ = preprocess_stock_data(stock_data)
    
    model = train_linear_regression_model(stock_symbol)
    
    # Predict the next day's price
    last_day_data = X.tail(1)
    predicted_price = model.predict(last_day_data)
    predicted_values_list = predicted_price.tolist()
    
    return {"predictions": predicted_values_list}

def prepare_data_for_regression(stock_data):
    X = stock_data.drop('Close', axis=1)  # Assuming you have other features in the dataset
    y = stock_data['Close']  # The target variable (stock close price)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def create_lstm_model():
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(60, 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=1))  # Output layer: 1 value (next day's price)
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def preprocess_data(stock_data, time_steps=60):
    x_data, y_data = [], []
    for i in range(time_steps, len(stock_data)):
        x_data.append(scaled_data[i-time_steps:i, 0])
        y_data.append(scaled_data[i, 0])
    
    x_data, y_data = np.array(x_data), np.array(y_data)
    
    # Reshaping the data to 3D for LSTM input (samples, timesteps, features)
    x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], 1))
    
    return x_data, y_data, scaler

# Function to train a simple LSTM model to predict stock prices
def predict_stock_lstm(stock_data):
    # Prepare data for LSTM model
    stock_data = stock_data[['Close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_data)
    
    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)  # Reshape for LSTM input
    
    # Build LSTM model
    model = keras.Sequential([
        layers.LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)),
        layers.LSTM(units=50, return_sequences=False),
        layers.Dense(units=1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=5, batch_size=32)
    
    # Predict using the model
    prediction = model.predict(X[-1].reshape(1, X.shape[1], 1))
    return scaler.inverse_transform(prediction)[0][0]

@app.post("/fetch-stock-data/")
async def fetch_stock_data(stock_request: StockRequest):
    stock_symbol = stock_request.stock_symbol
    try:
        stock_data = get_stock_data(stock_symbol)
        
        # Convert DataFrame to JSON serializable format (list of dictionaries)
        data = stock_data.tail(5).to_dict(orient='records')  # Convert to JSON-friendly format
        
        return {"message": f"Stock data for {stock_symbol} fetched successfully", "data": data}
    except ValueError as e:
        return {"error": str(e)}


@app.post("/fetch-twitter-data/")
async def fetch_twitter_data(stock_request: StockRequest):
    stock_symbol = stock_request.stock_symbol
    try:
        tweets = get_twitter_data(stock_symbol)
        sentiment = analyze_sentiment_twitter(tweets)
        return {"message": f"Twitter data for {stock_symbol} fetched successfully", "sentiment": sentiment}
    except Exception as e:
        return {"error": str(e)}

@app.post("/fetch-news-sentiment/")
async def fetch_news_sentiment(stock_request: StockRequest):
    stock_symbol = stock_request.stock_symbol
    try:
        news_data = get_alpha_vantage_news(stock_symbol)
        sentiment = analyze_sentiment_news(news_data)
        return {"message": f"News sentiment for {stock_symbol} fetched successfully", "sentiment": sentiment}
    except Exception as e:
        return {"error": str(e)}

@app.post("/predict-stock-arima/")
async def predict_stock_arima(stock_request: StockRequest):
    stock_symbol = stock_request.stock_symbol
    
    # Fetch stock data
    stock_data = get_stock_data(stock_symbol)
    
    # Fit ARIMA model (you can adjust the order based on your tuning)
    model = ARIMA(stock_data['Close'], order=(5, 1, 0))  # Example order (p,d,q)
    model_fit = model.fit()
    
    # Forecast the next day's closing price
    forecast = model_fit.forecast(steps=1)  # Forecast the next day (1 step ahead)
    
    # Return the predicted price
    predicted_price = forecast[0]
    
    return JSONResponse(content={"predicted_price": predicted_price})

# Use this in your route handler to preprocess data and train the model
@app.post("/predict-stock-lstm/")
async def predict_stock_lstm(stock_request: StockRequest):
    try:
        stock_symbol = stock_request.stock_symbol
        
        # Fetch stock data with retry logic
        stock_data, scaled_data, scaler = get_stock_data(stock_symbol)
        
        # Preprocess data for LSTM
        x_data, y_data = preprocess_data(stock_data, scaled_data)
        
        # Create LSTM model
        model = create_lstm_model(input_shape=(x_data.shape[1], 1))
        
        # Train the model
        model.fit(x_data, y_data, epochs=10, batch_size=32)
        
        # Predict for the next day
        predicted_price = model.predict(x_data[-1].reshape(1, 60, 1))
        
        # Inverse transform the prediction
        predicted_price = scaler.inverse_transform(predicted_price)
        
        return JSONResponse(content={"predicted_price": predicted_price[0][0]})
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)


@app.post("/predict-stock-linear-regression/")
async def predict_stock_linear_regression_endpoint(stock_request: StockRequest):
    stock_symbol = stock_request.stock_symbol
    try:
        predicted_price = predict_stock_linear_regression(stock_symbol)
        return {"message": f"Linear Regression prediction for {stock_symbol}", "predicted_price": predicted_price}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in prediction: {str(e)}")