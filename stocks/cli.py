from configparser import ConfigParser
import os
from pathlib import Path

import alpaca_trade_api as tradeapi
import click
from tensorflow.keras.models import load_model
import yaml

import stocks.stocks as stocks


@click.group(help=__doc__)
@click.option("--config", default='config/config.yml')
@click.pass_context
def cli(ctx, config):
    with open(config) as f:
        var_config = yaml.safe_load(f)
    key_id = var_config['key_id']
    secret_key = var_config['secret_key']
    ctx.obj = tradeapi.REST(key_id, secret_key)


@cli.command()
@click.pass_obj
@click.option("--symbol", required=True, type=str)
@click.option("--save-path", default="model.h5", type=str)
def trade_stocks(trade_api, symbol, save_path):

    data_df = stocks.get_trade_data(trade_api, symbol)

    formatted_df = stocks.format_data(data_df)

    X_train, X_test, y_train, y_test = stocks.create_training_data(formatted_df)

    model = stocks.train_model(formatted_df, X_train, X_test, y_train, y_test)

    model.save(f'{symbol}_model.h5')
    os.path.abspath(f'{symbol}_model.h5')

    model.save(f'{save_path}/{symbol}_model.h5')
    os.path.abspath(f'{save_path}/{symbol}_model.h5')




    error = stocks.analyze(formatted_df, model, X_test, y_test)

    dict_file = {'error': error, 'data_frame': formatted_df}
    with open(rf'{save_path}/{symbol}_info.yml', 'w') as file:
        yaml.dump(dict_file, file)

    stocks.traiding_test(formatted_df, model, error)


@cli.command()
@click.pass_obj
@click.option("--symbol", required=True, type=str)
@click.option("--upload-path", default="model.h5", type=str)
def buy_stocks_test(trade_api, symbol, upload_path):
    with open(f'{upload_path}/{symbol}_info.yml') as f:
        df_config = yaml.safe_load(f)
    formatted_df = df_config['data_frame']
    error = df_config['error']

    model = load_model(f'{upload_path}/{symbol}_model.h5')
    stocks.traiding_test(formatted_df, model, error)