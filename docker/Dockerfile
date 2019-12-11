FROM alectolytic/scipy
COPY . /code
RUN cd /code && \
    rm -rf .git && \
    python -m pip install --upgrade --no-cache-dir pip setuptools wheel && \
    python -m pip install --no-cache-dir -e .
