FROM python:3.7

WORKDIR /
COPY . /
RUN python setup.py develop

ENV key_id ""
ENV secret_key ""

ENTRYPOINT ["stocks"]