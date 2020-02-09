import pytest
import datetime

import pandas as pd
import pickle
import numpy as np
import yaml
from tensorflow.keras.models import load_model


@pytest.fixture
def alpaca_df():
    """Set up DataFrame to give to the method that will be tested."""

    # Set columns
    input_columns = ["day", "open", "high", "low", "close", "volume"]
    # Create data
    input_data = np.array(
        [
            [
                datetime.datetime(year=2018, month=1, day=1),
                18.01,
                18.1500,
                17.6300,
                18.02,
                15809912,
            ],
            [
                datetime.datetime(year=2019, month=1, day=1),
                18.12,
                18.4900,
                18.0500,
                18.35,
                8385317,
            ],
            [
                datetime.datetime(year=2020, month=1, day=1),
                18.00,
                18.1700,
                17.9600,
                18.02,
                7312304,
            ],
        ]
    )
    # Create DataFrame
    input_df = pd.DataFrame(data=input_data, columns=input_columns)
    # Set index to be day
    return input_df.set_index("day")


@pytest.fixture
def formatted_df():
    """Set up DataFrame as expected formatted data."""

    # set columns
    verify_columns = ["Year", "close", "1 days ago"]
    # Create data
    verify_data = np.array(
        [[2018.0, 18.02, 18.02], [2019.0, 18.35, 18.02], [2020.0, 18.02, 18.35]]
    )
    # Create DataFrame
    return pd.DataFrame(data=verify_data, columns=verify_columns)


@pytest.fixture
def test_df():
    """Load in the test formatted DataFrame."""

    return pickle.load(open("./stocks/test/TEST_files/TEST_model.pkl", "rb"))


@pytest.fixture
def test_model():
    """Load in the test trained TensorFlow model."""
    import os

    print(os.listdir("./"))
    return load_model("stocks/test/TEST_files/TEST_model.h5")


@pytest.fixture
def test_scaler():
    """Load in the test scaler"""

    return pickle.load(open("./stocks/test/TEST_files/TEST_scaler.pkl", "rb"))
