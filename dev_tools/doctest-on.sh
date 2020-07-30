#!/usr/bin/env bash

function path_to_module() {
  #1: source path
  local source_path
  source_path="${1}"
  if [[ ${source_path: -3} == ".py" ]]; then
    source_path="${source_path::-3}"
    if [[ ${source_path:0:4} == "src/" ]]; then
      source_path="${source_path:4}"
      # Replace "/" with "."
      source_path="${source_path////.}"
    fi
  fi
  echo "${source_path}"
}

function main() {
  #1: python module to run doctest on
  local source_path
  local module_path
  source_path="${1}"
  module_path="$(path_to_module "${source_path}")"
  python -c "import doctest; import pyhf; doctest.testmod(${module_path})"
  # Write check for this. Be optimistic now
  echo "# doctest succeeded!"
}

main "$@" || exit 1
