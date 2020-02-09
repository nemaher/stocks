from stocks import util
from unittest.mock import call, patch


@patch("stocks.util.print")
def test_progbar(mock_print):
    util.progbar(1, 2, 2)

    mock_print.assert_has_calls([call('\r', '#-', '[ 50.00%]', end='')])
