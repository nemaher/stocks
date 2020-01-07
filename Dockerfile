FROM python:3.7

WORKDIR /
COPY . /
RUN python setup.py develop

ENTRYPOINT ["stocks"]
CMD ["--help"]