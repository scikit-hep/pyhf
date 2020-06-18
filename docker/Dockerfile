ARG BASE_IMAGE=python:3.7-slim
# hadolint ignore=DL3006
FROM ${BASE_IMAGE} as base

FROM base as builder
COPY . /code
RUN apt-get -qq -y update && \
    apt-get -qq -y install --no-install-recommends \
        git && \
    apt-get -y autoclean && \
    apt-get -y autoremove && \
    rm -rf /var/lib/apt-get/lists/* && \
    cd /code && \
    python -m pip install --upgrade --no-cache-dir pip setuptools wheel && \
    python -m pip install --no-cache-dir .[xmlio] && \
    python -m pip list

FROM base
RUN apt-get -qq -y update && \
    apt-get -qq -y install --no-install-recommends \
        curl && \
    apt-get -y autoclean && \
    apt-get -y autoremove && \
    rm -rf /var/lib/apt-get/lists/*
COPY --from=builder /usr/local /usr/local
ENTRYPOINT ["/usr/local/bin/pyhf"]
