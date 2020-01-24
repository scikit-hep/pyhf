FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04 as base

FROM base as builder
ENV PYTHON_VERSION=cp36
ENV CUDA_VERSION=cuda101
ENV PLATFORM=linux_x86_64
ENV BASE_URL="https://storage.googleapis.com/jax-releases"
ENV JAXLIB_GPU_WHEEL="${BASE_URL}/${CUDA_VERSION}/jaxlib-0.1.37-${PYTHON_VERSION}-none-${PLATFORM}.whl"
RUN apt-get update -y && \
    apt-get install -y \
      git \
      python3 \
      python3-pip && \
    apt-get -y autoclean && \
    apt-get -y autoremove && \
    rm -rf /var/lib/apt-get/lists/*
COPY . /code
RUN cd /code && \
    python3 -m pip install --upgrade --no-cache-dir pip setuptools wheel && \
    python3 -m pip install --no-cache-dir .[xmlio] && \
    python3 -m pip install --no-cache-dir $JAXLIB_GPU_WHEEL && \
    python3 -m pip install --no-cache-dir jax && \
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
