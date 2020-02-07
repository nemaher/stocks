from stocks import stocks
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from collections import namedtuple


def test_get_training_data():
    """Test that if Exception is thrown when collecting historic data it is tried again."""

    DataFrame = namedtuple("DataFrame", "df")
    trade_api = MagicMock()
    trade_api.polygon.historic_agg.side_effect = [
        Exception("e"),
        Exception("e"),
        DataFrame(df="Test"),
    ]
    trade_data = stocks.get_trade_data(trade_api, "TEST")

    assert trade_data == "Test"


def test_format_data(alpaca_df, formatted_df):
    """Test that data is correctly formatted."""

    # run method to test
    output_df = stocks.format_data(alpaca_df, days=1)

    # verify that the return DataFrame is the same as the verify DataFrame
    assert output_df.equals(formatted_df)


def test_create_training_data(formatted_df):
    """Test that training data is split correctly."""

    X_train, X_test, y_train, y_test, scaler = stocks.create_training_data(formatted_df)

    # assert 2 data points
    assert len(X_train[0]) == 2
    print(X_train)
    assert len(X_test[0]) == 2
    print(X_test)

    # assert 1 test data point
    assert len(y_train) == 2  # 2 data points of data
    print(y_train)
    assert len(y_test) == 1
    print(y_test)

    # assert scaler is a MinMax 0-1 range
    assert str(scaler) == "MinMaxScaler(copy=True, feature_range=(0, 1))"


def test_train_model():
    """Test that a TensorFlow model is created."""

    test_model = stocks.train_model(
        2, [[1.0, 0.0], [0.0, 0.0]], [[2.0, 0.33]], [18.35, 18.02], [18.02]
    )

    assert "tensorflow.python.keras.engine.sequential.Sequential" in str(test_model)
