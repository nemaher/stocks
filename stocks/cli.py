from configparser import ConfigParser
import os
from pathlib import Path

import alpaca_trade_api as tradeapi
import click
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
    
    model.save(f'{save_path}/{symbol}_model.h5')

    error = stocks.analyze(formatted_df, model, X_test, y_test)

    stocks.traiding_test(formatted_df, model, error)


