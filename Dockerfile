FROM python:3.7

WORKDIR /
COPY . /
RUN python setup.py develop

ENTRYPOINT ["stocks"]
CMD ["--help"]

# CMD ["stocks", "--config", "config.cfg", "trade-stocks", "--symbol", "AAPL"]