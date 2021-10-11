ARG BASE_IMAGE=python:3.9-slim-bullseye
# hadolint ignore=DL3006
FROM ${BASE_IMAGE} as base

FROM base as builder
# Set PATH to pickup virtual environment by default
ENV PATH=/usr/local/venv/bin:"${PATH}"
COPY . /code
# hadolint ignore=DL3003,SC2102
RUN apt-get -qq -y update && \
    apt-get -qq -y install --no-install-recommends \
        git && \
    apt-get -y autoclean && \
    apt-get -y autoremove && \
    rm -rf /var/lib/apt/lists/* && \
    python -m venv /usr/local/venv && \
    cd /code && \
    python -m pip --no-cache-dir install --upgrade pip setuptools wheel && \
    python -m pip --no-cache-dir install .[xmlio,contrib] && \
    python -m pip list

FROM base
ENV PATH=/usr/local/venv/bin:"${PATH}"
RUN apt-get -qq -y update && \
    apt-get -qq -y install --no-install-recommends \
        curl && \
    apt-get -y autoclean && \
    apt-get -y autoremove && \
    rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/local/venv /usr/local/venv
ENTRYPOINT ["/usr/local/venv/bin/pyhf"]
