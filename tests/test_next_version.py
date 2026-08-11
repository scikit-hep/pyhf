import importlib.util
from pathlib import Path

import pytest

packaging_version = pytest.importorskip("packaging.version")

_spec = importlib.util.spec_from_file_location(
    "next_version",
    Path(__file__).parent.parent / ".github" / "scripts" / "next_version.py",
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
next_version = _module.next_version


@pytest.mark.parametrize(
    ("part", "release_candidate", "latest", "latest_stable", "expected"),
    [
        ("patch", False, "0.7.6", "0.7.6", "0.7.7"),
        ("minor", False, "0.7.6", "0.7.6", "0.8.0"),
        ("major", False, "0.7.6", "0.7.6", "1.0.0"),
        # first release candidate of a series
        ("patch", True, "0.7.6", "0.7.6", "0.7.7rc1"),
        ("minor", True, "0.7.6", "0.7.6", "0.8.0rc1"),
        ("major", True, "0.7.6", "0.7.6", "1.0.0rc1"),
        # increment the release candidate of the same target version
        ("minor", True, "0.8.0rc1", "0.7.6", "0.8.0rc2"),
        ("major", True, "1.0.0rc3", "0.7.6", "1.0.0rc4"),
        # a release candidate for a different target starts a new series
        ("patch", True, "0.8.0rc1", "0.7.6", "0.7.7rc1"),
        # a stable release finalizes a release candidate series
        ("minor", False, "0.8.0rc2", "0.7.6", "0.8.0"),
    ],
)
def test_next_version(part, release_candidate, latest, latest_stable, expected):
    assert (
        next_version(
            part,
            release_candidate,
            packaging_version.Version(latest),
            packaging_version.Version(latest_stable),
        )
        == expected
    )
