"""Validate that a release version is newer than the current release version."""

import argparse
from pathlib import Path

import tomllib
from packaging.version import Version


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "version", help="version of the release (e.g. 0.8.0 or 0.8.0rc1)"
    )
    args = parser.parse_args()

    with Path("tbump.toml").open("rb") as manifest:
        current_version = tomllib.load(manifest)["version"]["current"]
    if Version(args.version) <= Version(current_version):
        error_message = (
            f"ERROR: {args.version} is not newer than the current version"
            f" {current_version}."
            " Is this the correct branch for this release?"
        )
        raise SystemExit(error_message)
    print(f"Bumping version: {current_version} -> {args.version}")


if __name__ == "__main__":
    main()
