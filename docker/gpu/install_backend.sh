#!/bin/bash

set -e

function get_JAXLIB_GPU_WHEEL {
  # c.f. https://github.com/google/jax#pip-installation
  local PYTHON_VERSION # alternatives: cp35, cp36, cp37, cp38
  PYTHON_VERSION="cp"$(python3 --version | awk '{print $NF}' | awk '{split($0, rel, "."); print rel[1]rel[2]}')
  local CUDA_VERSION # alternatives: cuda90, cuda92, cuda100, cuda101
  CUDA_VERSION="cuda"$(< /usr/local/cuda/version.txt awk '{print $NF}' | awk '{split($0, rel, "."); print rel[1]rel[2]}')
  local PLATFORM=linux_x86_64
  local JAXLIB_VERSION=0.1.37
  local BASE_URL="https://storage.googleapis.com/jax-releases"
  local JAXLIB_GPU_WHEEL="${BASE_URL}/${CUDA_VERSION}/jaxlib-${JAXLIB_VERSION}-${PYTHON_VERSION}-none-${PLATFORM}.whl"
  echo "${JAXLIB_GPU_WHEEL}"
}

function install_backend() {
  # 1: the backend option name in setup.py
  local backend="${1}"
  if [[ "${backend}" == "tensorflow" ]]; then
    # shellcheck disable=SC2102
    python3 -m pip install --no-cache-dir .[xmlio,tensorflow]
  elif [[ "${backend}" == "torch" ]]; then
    # shellcheck disable=SC2102
    python3 -m pip install --no-cache-dir .[xmlio,torch]
  elif [[ "${backend}" == "jax" ]]; then
    python3 -m pip install --no-cache-dir .[xmlio]
    python3 -m pip install --no-cache-dir "$(get_JAXLIB_GPU_WHEEL)"
    python3 -m pip install --no-cache-dir jax
  fi
}

function main() {
  # 1: the backend option name in setup.py
  local BACKEND="${1}"
  install_backend "${BACKEND}"
}

main "$@" || exit 1
