import numpy as np
import pandas as pd
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Activation, Dense, Dropout
from tensorflow.keras.models import Sequential

scaler = MinMaxScaler()


def get_trade_data(trade_api, ticker_symbol):
    """Get the last 5 years of data for a given NY Stock Exchange ticker symbol.

    :param trade_api: Instance of the Alpaca API
    :param ticker_symbol: NY Stock Exchange ticker symbol to get data for
    :return: Pandas DataFrame of Historic data for the given ticker symbol
    """

    for i in range(3):
        try:
            # get Training data from alpaca api (1265 days is about 5 years)
            return trade_api.polygon.historic_agg_v2("day", ticker_symbol, limit=1265).df
        # If it failed try again
        except Exception as e:
            print(e)
            continue


def format_data(df, days=365):
    """Format the data so that each day has the year and amount of given days previous closing data associated with it.

    :param df: Data frame for a given ticker
    :param days: Set how many days back to get the data for
    :return: DataFrame with year and given days previous closing associated with each day.
    """

    #  replace day with just the year
    df = df.reset_index()
    df["day"] = pd.to_datetime(df["day"])
    df["Year"] = df["day"].apply(lambda date: date.year)

    #  Create a new data frame with historical data
    new_df = []
    print("\nProcessing data rows.")
    for index, row in df.iterrows():
        # create new row with year and current close data
        new_row = [row["Year"], row["close"]]
        for x in range(days + 1):
            if x != 0:
                # append past close data for amount of days specified
                new_row.append(df.iloc[index - x]["close"])
            else:
                continue

        # Add new row to data
        new_df.append(new_row)

    #  Set Columns
    columns = ["Year", "close"]
    for x in range(days + 1):
        if x != 0:
            columns.append(f"{x} days ago")
        else:
            continue

    #  convert numpy array to pandas DataFrame
    df = pd.DataFrame(data=np.array(new_df), columns=columns)
    return df


def create_training_data(df):
    """Creates Training and Test data to build the TensorFlow model.

    :param df: Formatted DataFrame to split testing and training data from
    :return:
        x_train = Day values to train data with.
        x_test = Close values (answers) of training data.
        y_train = Day values to test data with.
        y_test = Close values (answers) to test data.
        scaler = Scaler used to set values
    """

    # Remove close value for training data to keep it a secret
    X = df.drop("close", axis=1).values
    # Capture close data for test data
    y = df["close"].values

    # Split training data and testing data (30% test data, 70% training data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Use scaler to set values between 0 and 1
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


def train_model(node_num, X_train, X_test, y_train, y_test):
    """Train the model using TensorFlow.

    :param node_num: Number of nodes to add to each layer
    :param X_train: Day values to train data with from train_test_split.
    :param X_test: Close values (answers) of training data from train_test_split.
    :param y_train: Day values to test data with from train_test_split.
    :param y_test: Close values (answers) to test data from train_test_split.
    :return: Trained model
    """

    model = Sequential()

    # add layers to the neural network
    model.add(Dense(node_num, activation="relu"))
    model.add(Dropout(0.5))

    model.add(Dense(node_num, activation="relu"))
    model.add(Dropout(0.5))

    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    # Exit when data is becoming overfit for the data
    early_stop = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=15)

    # Train the model with the data
    model.fit(
        x=X_train,
        y=y_train,
        batch_size=128,
        epochs=1000,
        validation_data=(X_test, y_test),
        verbose=0,
        callbacks=[early_stop],
    )

    return model


def analyze(df, model, X_test, y_test):
    """Make a prediction from the model and calculate metrics.

    :param df: Original formatted DataFrame
    :param model: Trained TensorFlow model
    :param X_test: Test data points to make predictions with
    :param y_test: Test data answers to verify accuracy of predictions
    :return: Mean absolute error from data
    """

    # Predict stock price from test data
    predictions = model.predict(X_test)
    # calculate average error off from predictions
    error = round(mean_absolute_error(y_test, predictions), 2)
    print(f"Mean amount off ${error}")

    # calculate mean price of stock
    mean = df["close"].mean()
    print(f"Mean amount in data ${round(mean, 2)}")

    percent_off = error / mean
    print(f"Percent off from mean {round(percent_off*100, 2)}%")

    day_location = df.shape[0] - 1
    single_day = df.drop("close", axis=1).iloc[day_location]
    single_day = scaler.transform(single_day.values.reshape(-1, len(df.columns) - 1))

    # predicted price for day above
    predicted_price = round(model.predict(single_day)[0][0], 2)
    print(f"Predicted price using model ${str(predicted_price)}")

    # actual price for day above
    actual_price = round(df.iloc[day_location]["close"], 2)
    print(f"Actual price of day ${actual_price}")

    day_percent_off = round(predicted_price / actual_price, 2)
    day_amount_off = round(actual_price - predicted_price, 2)
    print(f"Day Amount off is ${day_amount_off} and Percent off is {day_percent_off}%")

    # Determine whether to buy or sell
    yesterdays_price = round(df.iloc[day_location - 1]["close"], 2)
    if predicted_price + error < yesterdays_price:
        print(
            f"SELL! Predicted price ${str(round(predicted_price + error, 2))} is less than Yesterdays price ${yesterdays_price}"
        )
    else:
        print(
            f"BUY! Predicted price ${str(round(predicted_price + error, 2))} is greater than Yesterdays price ${yesterdays_price}"
        )

    return error


def trading_test(df, model, error=0, scaler=scaler, money=500):
    """Predict metrics based off model and data if stocks were traded

    :param df: Original Formatted DataFrame
    :param model: Model created by TensorFlow to make predictions from
    :param error: Mean absolute error from data to add to price
    :param scaler: The scaler used to reshape test data
    :param money: Amount of starter money to trade with
    """

    amount_of_stock = 0

    # calculate trading if traded for last 900 days
    for x in reversed(range(900)):
        # 0 location is current day
        if x == 0:
            continue
        # get one day of data
        day_location = df.shape[0] - x
        single_day = df.drop("close", axis=1).iloc[day_location]
        single_day = scaler.transform(
            single_day.values.reshape(-1, len(df.columns) - 1)
        )

        # Predict todays price and get yesterdays price
        predicted_price = round(model.predict(single_day)[0][0], 2)
        yesterdays_price = round(df.iloc[day_location - 1]["close"], 2)

        # determine whether to buy or sell
        if predicted_price + error > yesterdays_price:
            while money > yesterdays_price:
                money = money - yesterdays_price
                amount_of_stock = amount_of_stock + 1
        else:
            while amount_of_stock > 0:
                money = money + yesterdays_price
                amount_of_stock = amount_of_stock - 1

    print(f"\n${round(money, 2)} left")
    print(f"{amount_of_stock} shares of stock")
    total_assets = round(amount_of_stock * yesterdays_price + money, 2)
    print(f"Total asset value is ${total_assets}")


def trade_stock(trade_api, ticker_symbol, df, model, error=0, scaler=scaler):
    """Determine predicted price of the stock and trade the stock with alpaca API

    :param trade_api: Alpaca API to make trades
    :param ticker_symbol: NY Stock Exchange ticker symbol to trade
    :param df: Formatted DataFrame to use to make predictions
    :param model: Trained Model to use to make predictions
    :param error: Mean absolute error from data to add to price
    :param scaler: scaler used to reshape test data
    """

    # get today's data to make a prediction of the closing price
    today = df.drop("close", axis=1).iloc[df.shape[0] - 1]
    today = scaler.transform(today.values.reshape(-1, len(df.columns) - 1))
    predicted_price = round(model.predict(today)[0][0], 2)

    # add error to predicted price for more accurate closing value
    est_price = predicted_price + error

    # only buy if price is 1% less than estimate max price
    buy_price = est_price - (est_price * 0.01)

    # actual price from alpaca API
    price = trade_api.polygon.snapshot(ticker_symbol).ticker["day"]["c"]

    # If price return from API is not 0 (market is open)
    if price != 0:

        print(
            f"Predicted price {predicted_price + error} greater than today price {price}. BUY"
        )
        try:
            trade_api.submit_order(
                symbol=ticker_symbol,
                qty=1,
                side="buy",
                type="market",
                time_in_force="day",
            )
        except Exception:
            return

        print(
            f"Predicted price {round(predicted_price, 2) + error} less than today price {price}. SELL"
        )
        try:
            trade_api.submit_order(
                symbol=ticker_symbol,
                qty=int(trade_api.get_position(ticker_symbol).qty),
                side="sell",
                type="market",
                time_in_force="day",
            )
        except Exception as err:
            print(err)
            return

        # Stock price is not 1% less than estimated price. Do not buy stock.
        else:
            print(
                f"Predicted price {est_price}, today price {price}. Stock price is {price/est_price}% less than estimated price. Did not buy or sell stock"
            )
