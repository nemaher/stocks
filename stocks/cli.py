from configparser import ConfigParser
import os
from pathlib import Path

import alpaca_trade_api as tradeapi
import click
import pickle
from tensorflow.keras.models import load_model
import yaml

import stocks.stocks as stocks


@click.group(help=__doc__)
@click.pass_context
def cli(ctx):
    # with open(config) as f:
    #     var_config = yaml.safe_load(f)
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

    model = stocks.train_model(formatted_df, X_train, X_test, y_train, y_test)

    error = stocks.analyze(formatted_df, model, X_test, y_test)

    formatted_df.to_pickle(f'{save_path}/{symbol}_model.pkl')
    
    model.save(f'{save_path}/{symbol}_model.h5')
    
    dict_file = {'error': int(error)}
    with open(rf'{save_path}/{symbol}_model.yml', 'w') as file:
        yaml.dump(dict_file, file)
        
    pickle.dump(scaler, open(f'{save_path}/{symbol}_scaler.pkl', 'wb'))
    
    stocks.traiding_test(formatted_df, model, error)


@cli.command()
@click.pass_obj
@click.option("--symbol", required=True, type=str)
@click.option("--upload-path", default="model.h5", type=str)
def buy_stocks_test(trade_api, symbol, upload_path):
    with open(f'{upload_path}/{symbol}_model.yml') as f:
        df_config = yaml.safe_load(f)

    formatted_df = pickle.load(open(f'{upload_path}/{symbol}_model.pkl', "rb"))
    model = load_model(f'{upload_path}/{symbol}_model.h5')
    error = df_config.get('error')
    scaler = pickle.load(open(f'{upload_path}/{symbol}_scaler.pkl', "rb"))

    stocks.traiding_test(formatted_df, model, error, scaler)
