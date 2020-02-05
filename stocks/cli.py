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
    key_id = os.environ['key_id']
    secret_key = os.environ['secret_key']
    ctx.obj = tradeapi.REST(key_id, secret_key)


@cli.command()
@click.pass_obj
@click.option("--symbols", required=True, type=str)
@click.option("--save-path", required=True, type=str)
def trade_stocks(trade_api, symbols, save_path):
    for symbol in symbols.split(","):
        print(f"Training for {symbol}")
        os.mkdir(f"{save_path}/{symbol}")
        symbol_save_path = f"{save_path}/{symbol}"
        data_df = stocks.get_trade_data(trade_api, symbol)

        formatted_df = stocks.format_data(data_df)

        X_train, X_test, y_train, y_test, scaler = stocks.create_training_data(formatted_df)

        low_error = 10000
        for _ in range(0, 5):
            model = stocks.train_model(formatted_df, X_train, X_test, y_train, y_test)

            error = stocks.analyze(formatted_df, model, X_test, y_test)

            if abs(error) < low_error:
                low_error = abs(error)

                formatted_df.to_pickle(f'{symbol_save_path}/{symbol}_model.pkl')

                model.save(f'{symbol_save_path}/{symbol}_model.h5')

                dict_file = {'error': float(error)}
                with open(rf'{symbol_save_path}/{symbol}_model.yml', 'w') as file:
                    yaml.dump(dict_file, file)

                pickle.dump(scaler, open(f'{symbol_save_path}/{symbol}_scaler.pkl', 'wb'))

        stocks.traiding_test(formatted_df, model, error)


@cli.command()
@click.pass_obj
@click.option("--upload-path", default="artifacts/stocks", type=str)
def buy_stocks(trade_api, upload_path):
    for directory in [dI for dI in os.listdir(upload_path) if os.path.isdir(os.path.join(upload_path, dI))]:
        print(f"Symbol {directory}")
        symbol = directory
        symbol_upload_path = f"{upload_path}/{directory}"

        for file in os.listdir(symbol_upload_path):
            if file.endswith("_model.yml"):
                with open(f'{symbol_upload_path}/{file}') as f:
                    df_config = yaml.safe_load(f)
                error = df_config.get('error')

            if file.endswith("_model.pkl"):
                formatted_df = pickle.load(open(f'{symbol_upload_path}/{file}', "rb"))

            if file.endswith("_model.h5"):
                model = load_model(f'{symbol_upload_path}/{file}')

            if file.endswith("_scaler.pkl"):
                scaler = pickle.load(open(f'{symbol_upload_path}/{file}', "rb"))

        stocks.trade_stock(trade_api, symbol, formatted_df, model, error, scaler)
