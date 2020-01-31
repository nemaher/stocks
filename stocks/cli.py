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
@click.option("--symbol", required=True, type=str)
@click.option("--save-path", required=True, type=str)
def trade_stocks(trade_api, symbol, save_path):
    data_df = stocks.get_trade_data(trade_api, symbol)

    formatted_df = stocks.format_data(data_df)

    X_train, X_test, y_train, y_test, scaler = stocks.create_training_data(formatted_df)

    low_error = 10000
    for _ in range(0, 5):
        model = stocks.train_model(formatted_df, X_train, X_test, y_train, y_test)

        error = stocks.analyze(formatted_df, model, X_test, y_test)

        if abs(error) < low_error:
            low_error = abs(error)

            formatted_df.to_pickle(f'{save_path}/{symbol}_model.pkl')

            model.save(f'{save_path}/{symbol}_model.h5')

            dict_file = {'error': float(error)}
            with open(rf'{save_path}/{symbol}_model.yml', 'w') as file:
                yaml.dump(dict_file, file)

            pickle.dump(scaler, open(f'{save_path}/{symbol}_scaler.pkl', 'wb'))

    stocks.traiding_test(formatted_df, model, error)


@cli.command()
@click.pass_obj
@click.option("--symbol", required=True, type=str)
@click.option("--upload-path", default="model.h5", type=str)
def buy_stocks(trade_api, symbol, upload_path):
    for file in os.listdir(upload_path):
        if file.endswith("_model.yml"):
            with open(f'{upload_path}/{file}') as f:
                df_config = yaml.safe_load(f)
            error = df_config.get('error')

        if file.endswith("_model.pkl"):
            formatted_df = pickle.load(open(f'{upload_path}/{file}', "rb"))

        if file.endswith("_model.h5"):
            model = load_model(f'{upload_path}/{file}')

        if file.endswith("_scaler.pkl"):
            scaler = pickle.load(open(f'{upload_path}/{file}', "rb"))

    stocks.trade_stock(trade_api, symbol, formatted_df, model, error, scaler)
