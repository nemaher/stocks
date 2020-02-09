def progbar(curr, total, full_progbar):
    """Display a progress bar for loading data.

    :param curr: current value
    :param total: total value
    :param full_progbar: value for full bar
    """

    frac = curr / total
    filled_progbar = round(frac * full_progbar)
    print(
        "\r",
        "#" * filled_progbar + "-" * (full_progbar - filled_progbar),
        "[{:>7.2%}]".format(frac),
        end="",
    )
