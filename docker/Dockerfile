ARG BASE_IMAGE=python:3.12-slim-bullseye
# hadolint ignore=DL3006
FROM ${BASE_IMAGE} AS base

FROM base AS builder
# Set PATH to pickup virtual environment by default
ENV PATH=/usr/local/venv/bin:"${PATH}"
COPY . /code
# hadolint ignore=DL3003,SC2102,SC1091
RUN apt-get -qq -y update && \
    apt-get -qq -y install --no-install-recommends \
        git && \
    apt-get -y autoclean && \
    apt-get -y autoremove && \
    rm -rf /var/lib/apt/lists/* && \
    python -m venv /usr/local/venv && \
    . /usr/local/venv/bin/activate && \
    cd /code && \
    python -m pip --no-cache-dir install --upgrade uv && \
    uv pip install --no-cache '.[xmlio,contrib]' && \
    uv pip list

FROM base

USER root

SHELL [ "/bin/bash", "-c" ]
ENV PATH=/usr/local/venv/bin:"${PATH}"

RUN apt-get -qq -y update && \
    apt-get -qq -y install --no-install-recommends \
        curl && \
    apt-get -y autoclean && \
    apt-get -y autoremove && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user "moby" with uid 1000
RUN adduser \
      --shell /bin/bash \
      --gecos "default user" \
      --uid 1000 \
      --disabled-password \
      moby && \
    chown -R moby /home/moby && \
    mkdir /work && \
    chown -R moby /work && \
    echo -e "\nexport PATH=/usr/local/venv/bin:${PATH}\n" >> /home/moby/.bashrc

COPY --from=builder --chown=moby /usr/local/venv /usr/local/venv/

# Does NOT define USER as ENV
USER moby

ENV USER=moby
ENV HOME=/home/moby
WORKDIR ${HOME}/work

# Use C.UTF-8 locale to avoid issues with ASCII encoding
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ENV PATH=${HOME}/.local/bin:${PATH}

# The first ever run of the CLI API incurs some overhead so do that during the
# build instead of making the user wait
RUN pyhf --version

ENTRYPOINT ["/usr/local/venv/bin/pyhf"]
