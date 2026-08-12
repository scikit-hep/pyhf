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

    with Path("tbump.toml").open("rb") as manifest:
        version_config = tomllib.load(manifest)["version"]

    # Validate with the tbump.toml version regex, which tbump compiles in
    # verbose mode, to fail with a clear error here before tbump runs
    # (packaging.version.Version would otherwise accept versions tbump
    # rejects, e.g. a leading "v")
    if not re.fullmatch(version_config["regex"], args.version, flags=re.VERBOSE):
        error_message = (
            f"ERROR: {args.version} does not match the tbump.toml release"
            " version format X.Y.Z or X.Y.ZrcN (with no leading v)."
        )
        raise SystemExit(error_message)

    # packaging normalizes versions (e.g. 0.8.00 to 0.8.0), so require the
    # canonical form to keep the released version identical everywhere
    if args.version != str(Version(args.version)):
        error_message = (
            f"ERROR: {args.version} is not the canonical form"
            f" {Version(args.version)} of the version."
        )
        raise SystemExit(error_message)

    current_version = version_config["current"]
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
