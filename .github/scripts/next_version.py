"""Compute the next release version from the Git tags reachable from HEAD."""

import argparse
import subprocess
from pathlib import Path

import tomllib
from packaging.version import Version


def next_version(part, release_candidate, latest, latest_stable):
    """
    Compute the next release version.

    Stable releases bump ``part`` relative to the latest stable release, so a
    stable release after a release candidate series finalizes it (e.g. latest
    tag v0.8.0rc2 with a minor bump gives v0.8.0). Release candidates increment
    the candidate number if the latest tag is already a candidate for the
    target version, and start at rc1 otherwise.

    Args:
        part (str): The semantic version part to bump: major, minor, or patch.
        release_candidate (bool): If the next version is a release candidate.
        latest (packaging.version.Version): The latest release, stable or not.
        latest_stable (packaging.version.Version): The latest stable release.

    Returns:
        str: The next version, without a leading "v".
    """
    major, minor, patch = latest_stable.release
    target = {
        "major": (major + 1, 0, 0),
        "minor": (major, minor + 1, 0),
        "patch": (major, minor, patch + 1),
    }[part]
    if not release_candidate:
        return "{}.{}.{}".format(*target)
    if latest.is_prerelease and latest.release == target:
        return "{}.{}.{}rc{}".format(*target, latest.pre[1] + 1)
    return "{}.{}.{}rc1".format(*target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=["major", "minor", "patch"], required=True)
    parser.add_argument(
        "--rc",
        choices=["true", "false"],
        default="false",
        help="if the next version is a release candidate",
    )
    args = parser.parse_args()

    # Only tags reachable from HEAD, so releases from a release/vX.Y.x branch
    # are computed relative to that release series and not the default branch.
    tags = subprocess.run(
        ["git", "tag", "--list", "v*", "--merged", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    versions = [Version(tag.removeprefix("v")) for tag in tags]
    latest = max(versions)
    latest_stable = max(version for version in versions if not version.is_prerelease)
    version = next_version(args.part, args.rc == "true", latest, latest_stable)

    # The patch release tags of a release series are only reachable from its
    # release/vX.Y.x branch, so guard against computing a version bump from a
    # branch that is not part of the intended release series (e.g. a patch
    # release of an old series attempted from the default branch).
    with Path("tbump.toml").open("rb") as manifest:
        current = Version(tomllib.load(manifest)["version"]["current"])
    if Version(version) <= current:
        error_message = (
            f"ERROR: computed next version {version} is not newer than the"
            f" current version {current} in tbump.toml."
            " Is this the correct branch for this release?"
        )
        raise SystemExit(error_message)

    print(version)


if __name__ == "__main__":
    main()
