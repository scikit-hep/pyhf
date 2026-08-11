"""Validate that a release version is newer than the current release version."""

import argparse
import re
from pathlib import Path

import tomllib
from packaging.version import Version


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "version", help="version of the release (e.g. 0.8.0 or 0.8.0rc1)"
    )
    args = parser.parse_args()

    # Same version format that the tbump.toml regex enforces, checked here to
    # fail with a clear error before tbump runs (packaging.version.Version
    # would otherwise accept versions tbump rejects, e.g. a leading "v")
    if not re.fullmatch(r"\d+\.\d+\.\d+(rc\d+)?", args.version):
        error_message = (
            f"ERROR: {args.version} does not match the release version format"
            " X.Y.Z or X.Y.ZrcN (with no leading v)."
        )
        raise SystemExit(error_message)

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
