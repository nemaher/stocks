import os

import alpaca_trade_api as tradeapi
import click
import pickle
from tensorflow.keras.models import load_model
import yaml

import stocks.stocks as stocks


@click.group(help=__doc__)
@click.pass_context
def cli(ctx):
    """Use environment variables "key_id" and "secret_key" to authenticate with alpaca

    :param ctx: Context to pass to methods
    """

    key_id = os.environ['key_id']
    secret_key = os.environ['secret_key']
    ctx.obj = tradeapi.REST(key_id, secret_key)


@cli.command()
@click.pass_obj
@click.option("--symbols", required=True, type=str)
@click.option("--save-path", required=True, type=str)
def train_model(trade_api, symbols, save_path):
    """Train the models and save output to use for stock predictions

    :param trade_api: Passed context of alpaca api object
    :param symbols: List of NY Stock Exchange symbols to train models
    :param save_path: Path to save created files to use for trading
    """

    # Get each symbol from list
    for symbol in symbols.split(","):
        print(f"Training for {symbol}")

        # Make a dir to save model and data
        os.mkdir(f"{save_path}/{symbol}")
        symbol_save_path = f"{save_path}/{symbol}"

        # Get training data from Alpaca API, format it and create test set
        data_df = stocks.get_trade_data(trade_api, symbol)
        formatted_df = stocks.format_data(data_df)
        X_train, X_test, y_train, y_test, scaler = stocks.create_training_data(formatted_df)

        low_error = 100000

        # Run data 5 5 times to get best trained model
        for _ in range(0, 5):

            # train model with formatted data and test set
            model = stocks.train_model(len(formatted_df.columns) - 1, X_train, X_test, y_train, y_test)

            # analyze data and get mean error predictions are off
            error = stocks.analyze(formatted_df, model, X_test, y_test)

            # If mean error is lower than last run set error as lowest error and save model data
            if abs(error) < low_error:
                low_error = abs(error)

                formatted_df.to_pickle(f'{symbol_save_path}/{symbol}_model.pkl')

                model.save(f'{symbol_save_path}/{symbol}_model.h5')

                dict_file = {'error': float(error)}
                with open(rf'{symbol_save_path}/{symbol}_model.yml', 'w') as file:
                    yaml.dump(dict_file, file)

                pickle.dump(scaler, open(f'{symbol_save_path}/{symbol}_scaler.pkl', 'wb'))

        # test model makes profit
        stocks.trading_test(formatted_df, model, error)


@cli.command()
@click.pass_obj
@click.option("--upload-path", default="artifacts/stocks", type=str)
def trade_stocks(trade_api, upload_path):
    """Predict stock price and determine whether to buy or sell

    :param trade_api: Passed context of alpaca api object
    :param upload_path: Path to get the ticker symbols and related models
    """

    # Get all directory names within the upload path
    for directory in [dI for dI in os.listdir(upload_path) if os.path.isdir(os.path.join(upload_path, dI))]:
        print(f"Symbol {directory}")
        symbol = directory
        symbol_upload_path = f"{upload_path}/{directory}"

        # import all files associated with that symbol
        for file in os.listdir(symbol_upload_path):
            # if file.endswith("_model.yml"):
            #     with open(f'{symbol_upload_path}/{file}') as f:
            #         df_config = yaml.safe_load(f)
            #     error = df_config.get('error')

            if file.endswith("_model.pkl"):
                formatted_df = pickle.load(open(f'{symbol_upload_path}/{file}', "rb"))

            if file.endswith("_model.h5"):
                model = load_model(f'{symbol_upload_path}/{file}')

            if file.endswith("_scaler.pkl"):
                scaler = pickle.load(open(f'{symbol_upload_path}/{file}', "rb"))


        # Trade the symbols stock
        stocks.trade_stock(trade_api, symbol, formatted_df, model, error=0, scaler=scaler)
