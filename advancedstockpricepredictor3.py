# Import necessary libraries
import yfinance as yf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf
from datetime import datetime
import csv

# Force TensorFlow to use CPU
tf.config.set_visible_devices([], 'GPU')

# List of Nifty 50 Stocks
nifty_50_stocks = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS",
    "BAJAJ-AUTO.NS", "BAJAJFINSV.NS", "BAJFINANCE.NS", "BHARTIARTL.NS", "BPCL.NS",
    "BRITANNIA.NS", "CIPLA.NS", "COALINDIA.NS", "DIVISLAB.NS", "DRREDDY.NS",
    "EICHERMOT.NS", "GRASIM.NS", "HCLTECH.NS", "HDFC.NS", "HDFCBANK.NS",
    "HDFCLIFE.NS", "HEROMOTOCO.NS", "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS",
    "INDUSINDBK.NS", "INFY.NS", "ITC.NS", "JSWSTEEL.NS", "KOTAKBANK.NS",
    "LT.NS", "M&M.NS", "MARUTI.NS", "NTPC.NS", "NESTLEIND.NS",
    "ONGC.NS", "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS", "SBIN.NS",
    "SUNPHARMA.NS", "TCS.NS", "TATACONSUM.NS", "TATAMOTORS.NS", "TATASTEEL.NS",
    "TECHM.NS", "TITAN.NS", "ULTRACEMCO.NS", "UPL.NS", "WIPRO.NS"
]

# Create CSV file for predictions
current_date = datetime.now().strftime("%Y-%m-%d")
csv_filename = f"stock_predictions_{current_date}.csv"

# Initialize CSV file with headers
with open(csv_filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Stock', 'Date', 'Predicted_Open', 'Predicted_High', 'Predicted_Low', 'Predicted_Close',
                     'Current_Open', 'Current_High', 'Current_Low', 'Current_Close'])


def prepare_lstm_data(data, feature_columns, lookback=7):
    """Prepare data for LSTM model with multiple features"""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[feature_columns])

    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i - lookback:i])
        y.append(scaled_data[i])
    return np.array(X), np.array(y), scaler


def build_lstm_model(lookback, n_features):
    """Build LSTM model for multiple feature prediction"""
    model = Sequential([
        LSTM(units=100, return_sequences=True, input_shape=(lookback, n_features)),
        LSTM(units=50, return_sequences=False),
        Dense(units=50),
        Dense(units=n_features)  # Predict all features
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def train_and_predict(stock_data, stock_symbol, lookback=120):
    """Train model and make predictions"""
    feature_columns = ['Open', 'High', 'Low', 'Close']

    # Prepare data
    X, y, scaler = prepare_lstm_data(stock_data, feature_columns, lookback)

    # Split data
    train_size = int(len(X) * 0.9)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Train model
    model = build_lstm_model(lookback, len(feature_columns))

    # Find optimal parameters
    best_loss = float('inf')
    optimal_params = None

    for batch_size in [32, 64]:
        for epochs in [50, 100]:
            history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                                validation_split=0.1, verbose=0)
            val_loss = min(history.history['val_loss'])

            if val_loss < best_loss:
                best_loss = val_loss
                optimal_params = (epochs, batch_size)

    # Final training with optimal parameters
    if optimal_params:
        epochs, batch_size = optimal_params
        print(f"\nTraining {stock_symbol} with optimal parameters: epochs={epochs}, batch_size={batch_size}")
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

        # Prepare sequence for prediction
        last_sequence = stock_data[feature_columns].values[-lookback:]
        scaled_sequence = scaler.transform(last_sequence)
        input_sequence = scaled_sequence.reshape(1, lookback, len(feature_columns))

        # Make predictions
        prediction = model.predict(input_sequence)
        predicted_values = scaler.inverse_transform(prediction)[0]

        # Get current values
        current_values = stock_data[feature_columns].iloc[-1]

        # Save predictions to CSV
        with open(csv_filename, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                stock_symbol,
                datetime.now().strftime("%Y-%m-%d"),
                round(predicted_values[0], 2),  # Predicted Open
                round(predicted_values[1], 2),  # Predicted High
                round(predicted_values[2], 2),  # Predicted Low
                round(predicted_values[3], 2),  # Predicted Close
                round(current_values['Open'], 2),
                round(current_values['High'], 2),
                round(current_values['Low'], 2),
                round(current_values['Close'], 2)
            ])

        # Plot predictions vs actual
        plot_predictions(stock_data, stock_symbol, model, X_test, y_test, scaler, feature_columns)

        return predicted_values, current_values
    return None, None


def plot_predictions(data, stock_symbol, model, X_test, y_test, scaler, feature_columns):
    """Plot actual vs predicted values for all price points"""
    predictions = model.predict(X_test)

    # Inverse transform predictions and actual values
    predictions = scaler.inverse_transform(predictions)
    actual = scaler.inverse_transform(y_test)

    # Create subplots for each price point
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{stock_symbol} - Price Predictions', fontsize=16)

    for idx, (col, ax) in enumerate(zip(feature_columns, axes.ravel())):
        ax.plot(actual[:, idx], label='Actual', color='blue')
        ax.plot(predictions[:, idx], label='Predicted', color='orange')
        ax.set_title(col)
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        ax.legend()

    plt.tight_layout()
    plt.show()


# Main execution
print("Starting price prediction for Nifty 50 stocks...")

for stock in nifty_50_stocks:
    try:
        print(f"\nProcessing {stock}...")

        # Fetch data
        stock_data = yf.Ticker(stock).history(period='5y')

        if len(stock_data) < 200:  # Skip if not enough data
            print(f"Insufficient data for {stock}, skipping...")
            continue

        # Clean data
        stock_data = stock_data.dropna()
        stock_data = stock_data[~stock_data.index.duplicated(keep='first')]

        # Train and predict
        predicted_values, current_values = train_and_predict(stock_data, stock)

        if predicted_values is not None:
            print(f"\nPredictions for {stock}:")
            print(f"Open:  Current: {current_values['Open']:.2f} → Predicted: {predicted_values[0]:.2f}")
            print(f"High:  Current: {current_values['High']:.2f} → Predicted: {predicted_values[1]:.2f}")
            print(f"Low:   Current: {current_values['Low']:.2f} → Predicted: {predicted_values[2]:.2f}")
            print(f"Close: Current: {current_values['Close']:.2f} → Predicted: {predicted_values[3]:.2f}")

    except Exception as e:
        print(f"Error processing {stock}: {str(e)}")

print(f"\nPredictions saved to {csv_filename}")