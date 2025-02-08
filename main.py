import os
import yfinance as yf
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.linear_model import LinearRegression
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from textblob import TextBlob
import gym
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
import zipfile
from io import BytesIO

# Initialize FastAPI app
app = FastAPI()

# Directory to store stock data
DATA_DIR = "data/stock_data/"
os.makedirs(DATA_DIR, exist_ok=True)
# Define the path to the static folder where the images are saved
STATIC_FOLDER = os.path.join(os.getcwd(), 'static')

# Serve static files (images) from the static folder
app.mount("/static", StaticFiles(directory="static"), name="static")


origins = [
    "http://localhost",  # Allow localhost (for development)
    "http://localhost:5173",  # React development server
    "http://localhost:5500",  # React development server
    "http://yourfrontenddomain.com",  # Add your frontend domain if it's deployed elsewhere
]

# Add CORSMiddleware to the app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Function to plot and save the graph for all models
def save_stock_price_plot(real_stock_price, predicted_stock_price, model_name="model"):
    """
    This function saves a plot of the actual vs predicted stock prices.
    
    Parameters:
    - real_stock_price: The actual stock prices.
    - predicted_stock_price: The predicted stock prices from the model.
    - model_name: The name of the model to be included in the saved file (e.g., 'LSTM', 'ARIMA', 'Linear Regression').
    """
    fig = plt.figure(figsize=(7.2, 4.8), dpi=65)
    
    # Plot actual vs predicted prices
    plt.plot(real_stock_price, label='Actual Price')
    plt.plot(predicted_stock_price, label='Predicted Price')
    
    # Add labels and legend
    plt.title(f"{model_name} - Actual vs Predicted Stock Prices")
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend(loc=4)
    
    # Save the figure
    file_name = f"static/{model_name}_stock_price_plot.png"
    plt.savefig(file_name)
    plt.close(fig)

# Pydantic model to get stock symbol from user
class StockRequest(BaseModel):
    stock_symbol: str

# Function to fetch stock data from Yahoo Finance
def get_stock_data(stock_symbol: str):
    stock_data = yf.download(stock_symbol, period="2y", interval="1d")
    if stock_data.empty:
        raise ValueError(f"No data found for symbol {stock_symbol}")
    return stock_data
    
def get_stock_data_arima(stock_symbol: str):
    # Fetch stock data from Yahoo Finance
    stock_data = yf.download(stock_symbol, period="2y", interval="1d")
    if stock_data.empty:
        raise ValueError(f"No data found for symbol {stock_symbol}")
    
    # Use only the 'Close' column for ARIMA
    stock_data = stock_data[['Close']].dropna()

    # Ensure the data has a DatetimeIndex with a frequency
    stock_data.index = pd.to_datetime(stock_data.index)
    stock_data = stock_data.asfreq('D', method='pad')  # 'D' means daily frequency
    
    return stock_data



def get_stock_data_lstm(stock_symbol: str):
    # Fetch stock data from Yahoo Finance
    stock_data = yf.download(stock_symbol, period="2y", interval="1d")
    if stock_data.empty:
        raise ValueError(f"No data found for symbol {stock_symbol}")
    
    # Use only the 'Close' column
    stock_data = stock_data[['Close']].dropna()
    
    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    stock_data_scaled = scaler.fit_transform(stock_data)
    
    # Prepare sequences for LSTM (e.g., 60 previous days to predict the next day)
    X, y = [], []
    for i in range(60, len(stock_data_scaled)):
        X.append(stock_data_scaled[i-60:i, 0])  # 60 time steps
        y.append(stock_data_scaled[i, 0])  # Target is the next day's close price
    
    X, y = np.array(X), np.array(y)
    
    # Reshape X for LSTM: [samples, time steps, features]
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    return X, y, scaler

def get_stock_data_linear_regression(stock_symbol: str):
    # Fetch stock data from Yahoo Finance
    stock_data = yf.download(stock_symbol, period="2y", interval="1d")
    if stock_data.empty:
        raise ValueError(f"No data found for symbol {stock_symbol}")
    
    # Use only the 'Close' column
    stock_data = stock_data[['Close']].dropna()
    
    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    stock_data_scaled = scaler.fit_transform(stock_data)
    
    # Split into X (features) and y (target)
    X = stock_data_scaled[:-1]  # All rows except the last
    y = stock_data_scaled[1:]   # All rows except the first
    
    return X, y, scaler

# Preprocess stock data for prediction
def preprocess_data(stock_data: pd.DataFrame):
    stock_data = stock_data[['Close']]  # Only using the 'Close' prices
    stock_data = stock_data.values  # Convert to numpy array
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    stock_data_scaled = scaler.fit_transform(stock_data)
    
    return stock_data_scaled, scaler

# Get news data from Alpha Vantage API
def get_alpha_vantage_news(stock_symbol: str):
    API_KEY = 'your_alpha_vantage_api_key_here'  # Replace with your API key
    url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={stock_symbol}&apikey={API_KEY}'
    response = requests.get(url)
    
    if response.status_code != 200:
        raise ValueError(f"Error fetching data from Alpha Vantage: {response.status_code}")
    
    data = response.json()
    news = [{'headline': article['title'], 'link': article['url']} for article in data.get('feed', [])]
    
    return news

# Analyze sentiment of news headlines
def analyze_sentiment_news(news_data):
    sentiment_scores = []
    for article in news_data:
        analysis = TextBlob(article['headline'])
        sentiment_scores.append(analysis.sentiment.polarity)
    return sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0

# Build ARIMA model
# Build ARIMA model
def build_arima_model(stock_data, forecast_steps=30):
    """
    This function builds the ARIMA model and returns the forecasted stock prices.
    """
    # Use only the 'Close' column for ARIMA
    stock_data = stock_data[['Close']].dropna()

    # Fit ARIMA model (example ARIMA(5, 1, 0) configuration)
    model = ARIMA(stock_data, order=(5, 1, 0))
    model_fit = model.fit()

    # Forecast stock prices (e.g., 30 days ahead)
    forecast = model_fit.forecast(steps=forecast_steps)

    # Plot the real vs predicted stock prices
    real_stock_price = stock_data['Close'].values[-forecast_steps:]  # Last 'forecast_steps' actual values
    save_stock_price_plot(real_stock_price, forecast, model_name="ARIMA")

    return forecast


# Build LSTM model
def build_lstm_model(X, y, scaler):
    # Reshaping X to 3D array if necessary (LSTM expects 3D input)
    X = X.reshape((X.shape[0], X.shape[1], 1))  # Ensure it's (samples, timesteps, features)
    
    # Create the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=1))  # Output layer
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X, y, epochs=10, batch_size=32)

    # Predict using the model
    predicted_stock_price = model.predict(X)

    # Inverse transform the predictions if a scaler is provided
    if scaler:
        predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
        real_stock_price = scaler.inverse_transform(y.reshape(-1, 1))

    # Call the save function to plot and save the graph
    save_stock_price_plot(real_stock_price, predicted_stock_price, model_name="LSTM")

    # Return the prediction
    return predicted_stock_price


# Build Linear Regression model
def build_linear_regression_model(X, y):
    # Fit the Linear Regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predict stock prices
    predicted_stock_price = model.predict(X)

    # Call the save function to plot and save the graph
    save_stock_price_plot(y, predicted_stock_price, model_name="Linear Regression")

    return predicted_stock_price


# Define a custom environment for stock trading
class StockTradingEnv(gym.Env):
    def __init__(self, stock_data, sentiment_data, initial_balance=1000):
        super(StockTradingEnv, self).__init__()

        self.stock_data = stock_data
        self.sentiment_data = sentiment_data
        self.initial_balance = initial_balance
        self.balance = self.initial_balance
        self.position = 0  # 0 = no position, 1 = holding a stock
        self.current_step = 0

        self.action_space = spaces.Discrete(3)  # 0 = Buy, 1 = Sell, 2 = Hold
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(stock_data[0]) + 1,), dtype=np.float32)

    def reset(self):
        self.balance = self.initial_balance
        self.position = 0
        self.current_step = 0
        return np.concatenate([self.stock_data[self.current_step], [self.sentiment_data[self.current_step]]])

    def step(self, action):
        prev_balance = self.balance
        prev_position = self.position

        # Action: 0 = Buy, 1 = Sell, 2 = Hold
        if action == 0 and self.balance >= self.stock_data[self.current_step]:  # Buy
            self.position = 1
            self.balance -= self.stock_data[self.current_step]
        elif action == 1 and self.position == 1:  # Sell
            self.position = 0
            self.balance += self.stock_data[self.current_step]
        
        # Move to next step (next day)
        self.current_step += 1
        done = self.current_step >= len(self.stock_data) - 1

        reward = self.balance - prev_balance
        return np.concatenate([self.stock_data[self.current_step], [self.sentiment_data[self.current_step]]]), reward, done, {}
    
# Your existing FastAPI routes go here
@app.get("/")
async def read_root():
    return {"message": "Hello World"}

# Define FastAPI route to make a decision
@app.post("/predict/")
async def predict_stock_action(stock_request: StockRequest):
    stock_symbol = stock_request.stock_symbol
    
    # Fetch stock data
    stock_data_arima = get_stock_data_arima(stock_symbol)
    X_lstm, y_lstm, scaler_lstm = get_stock_data_lstm(stock_symbol)
    X_lr, y_lr, scaler_lr = get_stock_data_linear_regression(stock_symbol)
    
    # ARIMA Prediction
    forecast_steps = 30  # Define the number of days to forecast
    arima_prediction = build_arima_model(stock_data_arima, forecast_steps=forecast_steps)

    # Get the last 'forecast_steps' actual stock prices for comparison
    real_stock_price_arima = stock_data_arima['Close'].values[-forecast_steps:]

    # Debugging: Print the ranges of real and predicted stock prices
    print("Real Stock Prices (Last 30 Days):", real_stock_price_arima)
    print("ARIMA Predicted Stock Prices:", arima_prediction)

    # Save the plot with real vs predicted prices
    save_stock_price_plot(real_stock_price_arima, arima_prediction, model_name="ARIMA")

    # LSTM Prediction
    lstm_prediction = build_lstm_model(X_lstm, y_lstm, scaler_lstm)
    # Reshape lstm_prediction to match dimensions
    lstm_prediction = lstm_prediction.reshape(-1)
    real_stock_price_lstm = scaler_lstm.inverse_transform(y_lstm.reshape(-1, 1)).reshape(-1)
    save_stock_price_plot(real_stock_price_lstm, lstm_prediction, model_name="LSTM")
    
    # Linear Regression Prediction
    lr_prediction = build_linear_regression_model(X_lr, y_lr)
    # Reshape lr_prediction to match dimensions
    lr_prediction = scaler_lr.inverse_transform(lr_prediction.reshape(-1, 1)).reshape(-1)
    real_stock_price_lr = scaler_lr.inverse_transform(y_lr.reshape(-1, 1)).reshape(-1)
    save_stock_price_plot(real_stock_price_lr, lr_prediction, model_name="Linear Regression")
    
    # Calculate average prediction (use last predicted value from each model)
    predictions = {
        "arima": float(arima_prediction[-1]),
        "lstm": float(lstm_prediction[-1]),
        "linear_regression": float(lr_prediction[-1])
    }
    
    current_price = float(stock_data_arima['Close'].iloc[-1])
    avg_prediction = float(np.mean(list(predictions.values())))
    
    # Decision logic
    decision = "Hold"
    if avg_prediction > current_price * 1.01:  # 1% threshold
        decision = "Buy"
    elif avg_prediction < current_price * 0.99:
        decision = "Sell"
    
    return JSONResponse(content={
        "predictions": predictions,
        "average": avg_prediction,
        "current_price": current_price,
        "decision": decision
    })


@app.get("/get-images")
async def get_images():
    """
    Return a zip file containing all the model images (e.g., LSTM, ARIMA, Linear Regression).
    The images should be saved as static/{model_name}_stock_price_plot.png.
    """
    model_names = ["LSTM", "ARIMA", "Linear Regression"]  # Add all model names you want here
    zip_buffer = BytesIO()

    # Create a zip file in memory
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for model_name in model_names:
            image_path = os.path.join(STATIC_FOLDER, f"{model_name}_stock_price_plot.png")
            if os.path.exists(image_path):
                zip_file.write(image_path, f"{model_name}_stock_price_plot.png")

    # Move the buffer's position to the beginning so it can be read
    zip_buffer.seek(0)

    # Return the zip file as a response with StreamingResponse
    return StreamingResponse(zip_buffer, media_type="application/zip", headers={"Content-Disposition": "attachment; filename=stock_price_plots.zip"})



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
