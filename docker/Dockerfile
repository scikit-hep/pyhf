FROM matthewfeickert/python3-scipy-alpine:scipy-1.2.1
COPY . /code
RUN cd /code && \
    apk add --no-cache git && \
    rm -rf /var/cache/apk/* && \
    python -m pip install --upgrade --no-cache-dir pip setuptools wheel && \
    python -m pip install --no-cache-dir -e .[xmlio] && \
    python -m pip list
ENTRYPOINT ["/usr/local/bin/pyhf"]
