from configparser import ConfigParser

import alpaca_trade_api as tradeapi
import click
import stocks.stocks as stocks

@click.group(help=__doc__)
@click.option("--config", required=True)
@click.pass_context
def cli(ctx, config):
    config_reader = ConfigParser()
    config_reader.read(config)
    key_id = config_reader.get("alpaca", "key_id")
    secret_key = config_reader.get("alpaca", "secret_key")
    ctx.obj = tradeapi.REST(key_id, secret_key)


@cli.command()
@click.pass_obj
@click.option("--symbol", required=True, type=str)
def trade_stocks(trade_api, symbol):

    data_df = stocks.get_trade_data(trade_api, symbol)

    formatted_df = stocks.format_data(data_df)

    X_train, X_test, y_train, y_test = stocks.create_training_data(formatted_df)

    model = stocks.train_model(formatted_df, X_train, X_test, y_train, y_test)

    error = stocks.analyze(formatted_df, model, X_test, y_test)

    stocks.traiding_test(formatted_df, model, error)