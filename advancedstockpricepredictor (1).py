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

# Import necessary libraries
import yfinance as yf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf
from datetime import datetime
current_date = datetime.now().strftime("%Y-%m-%d")
output_file = f"predicted_prices_{current_date}.txt"

file = open(output_file, "w")
# Force TensorFlow to use CPU
tf.config.set_visible_devices([], 'GPU')

# Check CPU usage
print("Using CPU for training and evaluation.")

# Fetch and clean historical data for the first 5 stocks
datalist = []
for stock in nifty_50_stocks[:5]:
    data = yf.Ticker(stock).history(period='max')
    data = data.dropna()  # Clean data
    data = data[~data.index.duplicated(keep='first')]
    data = data[data['Close'] > 0]
    if not data.empty:
        datalist.append((stock, data))


# Function to prepare data for LSTM
def prepare_lstm_data(data, lookback=7):
    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    # Create sequences
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i - lookback:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    return X, y, scaler


for stock, data in datalist:
    print(f"Training LSTM model for {stock}...")

    # Prepare data
    lookback = 120
    X, y, scaler = prepare_lstm_data(data, lookback=lookback)

    # Split into training and testing datasets
    train_size = int(len(X) * 0.9)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Reshape input data for LSTM [samples, time steps, features]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Loop through batch sizes and epochs to find optimal configuration
    target_accuracy = 0.999
    min_loss = 0.001
    optimal_epochs, optimal_batch_size = None, None
    for batch_size in [16, 32, 64]:  # Adjust as needed
        for epochs in [10, 20, 50, 100]:  # Adjust as needed
            # Build LSTM model
            model = Sequential([
                LSTM(units=50, return_sequences=True, input_shape=(lookback, 1)),
                LSTM(units=50, return_sequences=False),
                Dense(units=25),
                Dense(units=1)
            ])

            model.compile(optimizer='adam', loss='mean_squared_error')

            # Train the model
            history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
            loss = history.history['loss'][-1]

            # Evaluate on test data
            test_loss = model.evaluate(X_test, y_test, verbose=0)

            print(f"Batch Size: {batch_size}, Epochs: {epochs}, Loss: {test_loss}")

            if test_loss <= min_loss or 1 - test_loss >= target_accuracy:
                optimal_epochs = epochs
                optimal_batch_size = batch_size
                break
        if optimal_epochs and optimal_batch_size:
            break

    # Final training with optimal parameters
    if optimal_epochs and optimal_batch_size:
        print(f"Optimal Parameters for {stock} -> Epochs: {optimal_epochs}, Batch Size: {optimal_batch_size}")
        model.fit(X_train, y_train, epochs=optimal_epochs, batch_size=optimal_batch_size, verbose=1)

        # Predict on test data
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)  # Scale back to original values

        # Plot actual vs predicted using Seaborn
        # Scale back predictions and actual values to original
        actual = scaler.inverse_transform(y_test.reshape(-1, 1))  # Scale back to original values

        # Extract corresponding dates from the data index for test set
        test_dates = data.index[-len(y_test):]

        # Predict today's and tomorrow's prices
        last_sequence = data['Close'].values[-lookback:]  # Last 'lookback' days of actual prices
        scaled_last_sequence = scaler.transform(last_sequence.reshape(-1, 1))  # Scale to [0, 1]
        input_sequence = scaled_last_sequence.reshape(1, lookback, 1)  # Reshape for LSTM

        # Predict today's price
        today_prediction = model.predict(input_sequence)
        today_prediction_price = scaler.inverse_transform(today_prediction)[0, 0]  # Scale back

        # Predict tomorrow's price
        scaled_last_sequence = np.append(scaled_last_sequence[1:], today_prediction).reshape(-1, 1)  # Update sequence
        input_sequence = scaled_last_sequence.reshape(1, lookback, 1)  # Reshape for LSTM
        tomorrow_prediction = model.predict(input_sequence)
        tomorrow_prediction_price = scaler.inverse_transform(tomorrow_prediction)[0, 0]  # Scale back

        print(f"Predicted Price for Today ({stock}): {today_prediction_price:.2f} INR")
        print(f"Predicted Price for Tomorrow ({stock}): {tomorrow_prediction_price:.2f} INR")

        file.write(f"{stock} : {tomorrow_prediction_price - today_prediction_price}")
        file.write("\n")

        # Create a DataFrame for Seaborn
        import pandas as pd
        df = pd.DataFrame({
            'Date': test_dates,
            'Actual Prices': actual.flatten(),
            'Predicted Prices': predictions.flatten()
        })

        # Seaborn plot
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df, x='Date', y='Actual Prices', label="Actual Prices", color="blue")
        sns.lineplot(data=df, x='Date', y='Predicted Prices', label="Predicted Prices", color="orange")
        plt.title(f"{stock} - LSTM Predictions", fontsize=16)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Closing Price (INR)", fontsize=12)
        plt.legend(fontsize=10)
        plt.xticks(rotation=45)  # Rotate date labels for better readability
        plt.show()
    else:
        print(f"Failed to reach target accuracy for {stock}")

file.close()