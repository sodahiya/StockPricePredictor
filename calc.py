import pandas as pd
import yfinance as yf
import time

# File paths
csv_file = "2024-11-24.csv"  # Replace with your CSV file path
account_file = "dummy_account.txt"  # Replace with your dummy account file path

# Define target percentages
target_percentage_buy = 90  # 90% increase threshold for buying
target_percentage_sell = 120  # 120% increase threshold for selling


# Read the account file
def read_account(file_path):
    account_data = {"Starting Balance": 0, "Current Holdings": {}}
    try:
        with open(file_path, "r", encoding='utf-8') as file:
            lines = file.readlines()
    except UnicodeDecodeError:
        with open(file_path, "r") as file:
            lines = file.readlines()

    for line in lines:
        if line.startswith("Account Name:"):
            account_data["Account Name"] = line.split(":", 1)[1].strip()
        elif line.startswith("Starting Balance:"):
            value_str = line.split(":")[1].strip()
            value_str = value_str.replace("INR", "").replace(",", "").strip()
            account_data["Starting Balance"] = float(value_str)
        elif line.startswith("Total Current Stocks Net Worth:"):
            value_str = line.split(":")[1].strip()
            value_str = value_str.replace("INR", "").replace(",", "").strip()
            account_data["Stocks Net Worth"] = float(value_str)
        elif line.strip().startswith("Stock"):
            break
    else:
        return account_data

    holdings_start = lines.index("# Current Holdings\n") + 2
    for line in lines[holdings_start:]:
        if line.strip() == "":
            break
        parts = line.split()
        stock = parts[0]
        quantity = int(parts[1])
        price_str = parts[-2].replace("INR", "").strip()
        current_price = float(price_str)
        account_data["Current Holdings"][stock] = {"Quantity": quantity, "Current Price": current_price}

    return account_data


# Fetch live stock price using yfinance
def get_live_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        return stock.history(period="1d")["Close"].iloc[-1]
    except Exception as e:
        print(f"Error fetching live price for {ticker}: {e}")
        return None


# Analyze stocks for promising candidates
def analyze_stocks(df):
    promising_stocks = []
    for _, row in df.iterrows():
        stock = row['Stock']
        predicted_high = row['Predicted_High']
        current_low = row['Current_Low']

        change_percentage = ((predicted_high - current_low) / current_low) * 100

        if change_percentage >= target_percentage_buy:
            promising_stocks.append({"Stock": stock, "Change_Percentage": change_percentage})

    promising_stocks.sort(key=lambda x: x["Change_Percentage"], reverse=True)
    return promising_stocks[:4]


# Buy stocks with allocated capital
def buy_stocks(account_data, promising_stocks, allocated_capital):
    for stock_data in promising_stocks:
        stock = stock_data["Stock"]
        live_price = get_live_price(stock)
        if live_price:
            quantity = int(allocated_capital / live_price)
            if quantity > 0:
                cost = quantity * live_price
                account_data["Starting Balance"] -= cost
                if stock in account_data["Current Holdings"]:
                    account_data["Current Holdings"][stock]["Quantity"] += quantity
                else:
                    account_data["Current Holdings"][stock] = {"Quantity": quantity, "Current Price": live_price}
                print(f"Bought {quantity} shares of {stock} at INR {live_price:.2f} for a total of INR {cost:.2f}")


# Sell stocks if the live price exceeds the sell threshold
def sell_stocks(account_data):
    for stock, details in list(account_data["Current Holdings"].items()):
        live_price = get_live_price(stock)
        if live_price:
            purchase_price = details["Current Price"]
            change_percentage = ((live_price - purchase_price) / purchase_price) * 100

            if change_percentage >= target_percentage_sell:
                quantity = details["Quantity"]
                sale_value = quantity * live_price
                account_data["Starting Balance"] += sale_value
                del account_data["Current Holdings"][stock]
                print(f"Sold {quantity} shares of {stock} at INR {live_price:.2f} for a total of INR {sale_value:.2f}")


# Save updated account to file
def save_account(file_path, account_data):
    lines = [
        f"Account Name: {account_data['Account Name']}\n",
        f"Starting Balance: INR {account_data['Starting Balance']:,.2f}\n\n",
        f"Total Current Stocks Net Worth: INR {account_data.get('Stocks Net Worth', 0):,.2f}\n\n",
        "# Current Holdings\n",
        "Stock          Quantity      Current Price\n"
    ]
    for stock, details in account_data["Current Holdings"].items():
        lines.append(f"{stock}    {details['Quantity']}          INR {details['Current Price']:.2f}\n")

    try:
        with open(file_path, "w", encoding='utf-8') as file:
            file.writelines(lines)
    except Exception as e:
        print(f"Warning: Could not save with UTF-8 encoding ({str(e)})")
        with open(file_path, "w") as file:
            file.writelines(lines)


# Main workflow
def main():
    stock_data = pd.read_csv(csv_file)
    account_data = read_account(account_file)

    total_capital = account_data["Starting Balance"]
    allocated_capital = (total_capital * 4) / 5

    while True:
        print("Checking live stock prices...")

        promising_stocks = analyze_stocks(stock_data)
        per_stock_capital = allocated_capital / 4

        buy_stocks(account_data, promising_stocks, per_stock_capital)
        sell_stocks(account_data)
        save_account(account_file, account_data)

        print("Account updated! Waiting for 60 seconds before the next check.")
        time.sleep(60)


if __name__ == "__main__":
    main()
