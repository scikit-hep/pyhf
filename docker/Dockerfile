ARG BASE_IMAGE=python:3.8-slim
# hadolint ignore=DL3006
FROM ${BASE_IMAGE} as base

FROM base as builder
COPY . /code
# hadolint ignore=DL3003
RUN apt-get -qq -y update && \
    apt-get -qq -y install --no-install-recommends \
        git && \
    apt-get -y autoclean && \
    apt-get -y autoremove && \
    rm -rf /var/lib/apt/lists/* && \
    cd /code && \
    python -m pip --no-cache-dir install --upgrade pip setuptools wheel && \
    python -m pip --no-cache-dir install .[xmlio,contrib] && \
    python -m pip list

FROM base
RUN apt-get -qq -y update && \
    apt-get -qq -y install --no-install-recommends \
        curl && \
    apt-get -y autoclean && \
    apt-get -y autoremove && \
    rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/local /usr/local
ENTRYPOINT ["/usr/local/bin/pyhf"]
