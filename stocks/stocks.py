import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from stocks.util import progbar

scaler = MinMaxScaler()

def get_trade_data(trade_api, ticker_symbol):
    return trade_api.polygon.historic_agg('day', ticker_symbol, limit=1265).df

# Apple
# df = pd.read_csv('https://raw.githubusercontent.com/nemaher/TensorFlow/master/TF_2_Notebooks_and_Data/AAPL_5_Year.csv')

# Aurora
#df = pd.read_csv('https://raw.githubusercontent.com/nemaher/TensorFlow/master/TF_2_Notebooks_and_Data/ACB.csv')

# Tesla
#df = pd.read_csv('https://raw.githubusercontent.com/nemaher/TensorFlow/master/TF_2_Notebooks_and_Data/TSLA.csv')

# SEARS
# df = pd.read_csv('https://raw.githubusercontent.com/nemaher/TensorFlow/master/TF_2_Notebooks_and_Data/SHLDQ.csv')

# Canopy
# df = pd.read_csv('https://raw.githubusercontent.com/nemaher/TensorFlow/master/TF_2_Notebooks_and_Data/CGC.csv')

def format_data(df, days=365):
    #  days = Set how many days back to get the data for

    #  replace day with just the year
    df = df.reset_index()
    df['day'] = pd.to_datetime(df['day'])
    df['Year'] = df['day'].apply(lambda date: date.year)

    #  Create a new data frame with historical data 
    new_df = []
    print("\nProcessing data rows.")
    for index, row in df.iterrows():
        progbar(index, df.shape[0], 20)
        # create new row with year and current close data
        new_row = [row['Year'], row['close']]
        for x in range(days + 1):
            try:
                if x != 0:
                    # append past close data for amount of days specified
                    new_row.append(df.iloc[index - x]['close'])
                else:
                    continue
            except Exception:
                pass

        # Add new row to data
        new_df.append(new_row)

    #  Set Columns
    columns = ['Year', 'close']
    for x in range(days + 1):
        if x != 0:
            columns.append(f"{x} days ago")
        else:
            continue

    #  convert numpy array to pandas dataframe
    df = pd.DataFrame(data=np.array(new_df), columns=columns)
    return df

def create_training_data(df):
    X = df.drop('close', axis=1).values
    y = df['close'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def train_model(df, X_train, X_test, y_train, y_test):
    model = Sequential()

    model.add(Dense(len(df.columns)-1,activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(len(df.columns)-1,activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))

    model.compile(optimizer='adam',loss='mse')

    early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)

    model.fit(x=X_train,
              y=y_train,
              batch_size=128,
              epochs=1000,
              validation_data=(X_test, y_test), verbose=1,
              callbacks=[early_stop]
              )

    return model

# model_loss = pd.DataFrame(model.history.history)
# model_loss.plot()

def analyze(df, model, X_test, y_test):
    predictions = model.predict(X_test)
    error = round(mean_absolute_error(y_test,predictions), 2)
    print(f"Mean amount off ${error}")

    mean = df['close'].mean()
    print(f"Mean amount in data ${round(mean, 2)}")

    percent_off = error/mean
    print(f"Percent off from mean {round(percent_off*100, 2)}%")

    # # Our predictions
    # plt.scatter(y_test,predictions)
    #
    # # Perfect predictions
    # plt.plot(y_test,y_test,'r')

    day_location = df.shape[0] - 1
    # day_location = 2000
    single_day = df.drop('close',axis=1).iloc[day_location]
    single_day = scaler.transform(single_day.values.reshape(-1, len(df.columns)-1))

    # pridected price for day above
    pridected_price = round(model.predict(single_day)[0][0], 2)
    print(f"Pridected price using modle ${str(pridected_price)}")

    # actual price for day above
    actual_price = round(df.iloc[day_location]['close'], 2)
    print(f"Actual price of day ${actual_price}")

    day_percent_off = round(pridected_price/actual_price, 2)
    day_amount_off = round(actual_price - pridected_price, 2)
    print(f"Day Amount off is ${day_amount_off} and Percent off is {day_percent_off}%")

    yesterdays_price = round(df.iloc[day_location-1]['close'], 2)
    if pridected_price + error < yesterdays_price:
      print(f"SELL! Pridicted price ${str(round(pridected_price + error, 2))} is less than Yesterdays price ${yesterdays_price}")
    else:
      print(f"BUY! Pridicted price $({str(round(pridected_price + error, 2))} is greater than Yesterdays price ${yesterdays_price}")

    return error


def traiding_test(df, model, error):
    money = 500
    amount_of_stock = 0
    risky = False

    if risky:
      error = error
    else:
      error = 0

    for x in reversed(range(900)):
      # progbar(x, 900, 20)
      if x == 0:
          continue
      day_location = df.shape[0] - x
      single_day = df.drop('close', axis=1).iloc[day_location]
      single_day = scaler.transform(single_day.values.reshape(-1, len(df.columns) - 1))

      pridected_price = round(model.predict(single_day)[0][0], 2)
      yesterdays_price = round(df.iloc[day_location - 1]['close'], 2)

      if pridected_price + error > yesterdays_price:
          # print(f"pridected price {pridected_price + error} \n yesterday price {yesterdays_price}")
          while money > yesterdays_price:
              # print(f"Buy {x} days ago")
              money = money - yesterdays_price
              amount_of_stock = amount_of_stock + 1
      else:
          # print(f"day ago {x} amount of stock {amount_of_stock}")
          while amount_of_stock > 0:
              # print(f"Sell {x} days ago")
              money = money + yesterdays_price
              amount_of_stock = amount_of_stock - 1

    print(f"\n${round(money, 2)} left")
    print(f"{amount_of_stock} shares of stock")
    total_assets = round(amount_of_stock * yesterdays_price + money, 2)
    print(f"Total asset value is ${total_assets}")