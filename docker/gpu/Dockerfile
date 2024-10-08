FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04 AS base

FROM base AS builder
# hadolint ignore=DL3015
RUN apt-get update -y && \
    apt-get install -y \
      git \
      python3 \
      python3-pip && \
    apt-get -y autoclean && \
    apt-get -y autoremove && \
    rm -rf /var/lib/apt/lists/*
COPY . /code
COPY ./docker/gpu/install_backend.sh /code/install_backend.sh
WORKDIR /code
ARG BACKEND=tensorflow
RUN python3 -m pip --no-cache-dir install --upgrade pip wheel && \
    /bin/bash install_backend.sh ${BACKEND} && \
    python3 -m pip list

FROM base
# Use C.UTF-8 locale to avoid issues with ASCII encoding
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
COPY --from=builder /lib/x86_64-linux-gnu /lib/x86_64-linux-gnu
COPY --from=builder /usr/local /usr/local
COPY --from=builder /usr/bin/python3 /usr/bin/python3
COPY --from=builder /usr/bin/python3.6 /usr/bin/python3.6
COPY --from=builder /usr/bin/pip3 /usr/bin/pip3
COPY --from=builder /usr/lib/python3 /usr/lib/python3
COPY --from=builder /usr/lib/python3.6 /usr/lib/python3.6
COPY --from=builder /usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu
ENTRYPOINT ["/usr/local/bin/pyhf"]
