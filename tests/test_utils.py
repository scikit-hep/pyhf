import importlib.metadata
import io
import pathlib
import platform
import sys

import pytest

import pyhf


@pytest.mark.parametrize(
    ("opts", "obj"),
    [
        (["a=10"], {"a": 10}),
        (["b=test"], {"b": "test"}),
        (["c=1.0e-8"], {"c": 1.0e-8}),
        (["d=3.14"], {"d": 3.14}),
        (["e=True"], {"e": True}),
        (["f=false"], {"f": False}),
        (["a=b", "c=d"], {"a": "b", "c": "d"}),
        (["g=h=i"], {"g": "h=i"}),
    ],
)
def test_options_from_eqdelimstring(opts, obj):
    assert pyhf.utils.options_from_eqdelimstring(opts) == obj


@pytest.mark.parametrize(
    "obj",
    [
        {"a": 2.0, "b": 1.0, "c": "a"},
        {"b": 1.0, "c": "a", "a": 2.0},
        {"c": "a", "a": 2.0, "b": 1.0},
    ],
)
@pytest.mark.parametrize("algorithm", ["md5", "sha256"])
def test_digest(obj, algorithm):
    results = {
        "md5": "155e52b05179a1106d71e5e053452517",
        "sha256": "03dfbceade79855fc9b4e4d6fbd4f437109de68330dab37c3091a15f4bffe593",
    }
    assert pyhf.utils.digest(obj, algorithm=algorithm) == results[algorithm]


def test_digest_bad_obj():
    with pytest.raises(ValueError, match="not JSON-serializable"):
        pyhf.utils.digest(object())


def test_digest_bad_alg():
    with pytest.raises(ValueError, match="nonexistent_algorithm"):
        pyhf.utils.digest({}, algorithm="nonexistent_algorithm")


@pytest.mark.parametrize("oneline", [False, True])
def test_citation(oneline):
    citation = pyhf.utils.citation(oneline)
    assert citation
    if oneline:
        assert "\n" not in citation


def test_environment_info():
    info = pyhf.utils.environment_info()
    assert isinstance(info, str)
    assert "* os version:" in info
    assert "* kernel version:" in info
    assert "* python version:" in info
    assert f"* pyhf version: {pyhf.__version__}" in info
    assert "* numpy version:" in info
    assert "* scipy version:" in info
    assert "* iminuit version:" in info
    assert "* jax version:" in info
    assert "* jaxlib version:" in info
    # Output should be markdown bullet list lines
    for line in info.strip().splitlines():
        assert line.startswith("* ")


def test_environment_info_linux(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        platform,
        "freedesktop_os_release",
        lambda: {"NAME": "Ubuntu", "VERSION": "22.04.2 LTS (Jammy Jellyfish)"},
        raising=False,
    )
    info = pyhf.utils.environment_info()
    assert "* os version: Ubuntu 22.04.2 LTS (Jammy Jellyfish)" in info


def test_environment_info_linux_no_version(monkeypatch):
    # VERSION is optional in the os-release spec (e.g. rolling releases)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        platform,
        "freedesktop_os_release",
        lambda: {"NAME": "Arch Linux", "ID": "arch", "PRETTY_NAME": "Arch Linux"},
        raising=False,
    )
    info = pyhf.utils.environment_info()
    assert "* os version: Arch Linux" in info


def test_environment_info_linux_oserror(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")

    def raise_oserror():
        raise OSError

    monkeypatch.setattr(
        platform, "freedesktop_os_release", raise_oserror, raising=False
    )
    info = pyhf.utils.environment_info()
    assert "* os version: Cannot be determined" in info


def test_environment_info_linux_fallback_no_os_release_file(monkeypatch):
    # Simulate Python 3.9 (no platform.freedesktop_os_release) with no os-release file
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delattr(platform, "freedesktop_os_release", raising=False)

    original_open = pathlib.Path.open

    def mock_open(self, *args, **kwargs):
        if "os-release" in str(self):
            raise FileNotFoundError(str(self))
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "open", mock_open)
    info = pyhf.utils.environment_info()
    assert "* os version: Cannot be determined" in info


def test_environment_info_linux_fallback_parses_os_release(monkeypatch):
    # Simulate Python 3.9 with an os-release file exercising what the spec
    # permits: comment lines, quoted values, values containing "=", and no
    # NAME/VERSION so the reported value is PRETTY_NAME, whose embedded "="
    # pins that values are only split on the first "="
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delattr(platform, "freedesktop_os_release", raising=False)

    os_release_content = (
        "# This file is part of systemd\n"
        'PRETTY_NAME="Arch Linux (build=rolling)"\n'
        "ID=arch\n"
        'BUG_REPORT_URL="https://bugs.example.org/?product=arch"\n'
        "\n"
    )

    original_open = pathlib.Path.open

    def mock_open(self, *args, **kwargs):
        if "os-release" in str(self):
            return io.StringIO(os_release_content)
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "open", mock_open)
    info = pyhf.utils.environment_info()
    assert "* os version: Arch Linux (build=rolling)" in info


def test_environment_info_linux_version_id_fallback(monkeypatch):
    # Distributions like Alpine provide VERSION_ID but no VERSION
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        platform,
        "freedesktop_os_release",
        lambda: {
            "NAME": "Alpine Linux",
            "ID": "alpine",
            "VERSION_ID": "3.20.1",
            "PRETTY_NAME": "Alpine Linux v3.20",
        },
        raising=False,
    )
    info = pyhf.utils.environment_info()
    assert "* os version: Alpine Linux 3.20.1" in info


def test_environment_info_darwin(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(platform, "mac_ver", lambda: ("14.1.0", ("", "", ""), ""))
    info = pyhf.utils.environment_info()
    assert "* os version: macOS 14.1.0" in info


def test_environment_info_unknown_platform(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    info = pyhf.utils.environment_info()
    assert "* os version: Cannot be determined" in info


def test_environment_info_missing_optional(monkeypatch):
    original_version = importlib.metadata.version

    def mock_version(name):
        if name in ("iminuit", "jax", "jaxlib"):
            raise importlib.metadata.PackageNotFoundError(name)
        return original_version(name)

    monkeypatch.setattr(importlib.metadata, "version", mock_version)
    info = pyhf.utils.environment_info()
    assert "* iminuit version: not installed" in info
    assert "* jax version: not installed" in info
    assert "* jaxlib version: not installed" in info
    # Core packages still report their real versions, not "not installed"
    assert f"* numpy version: {original_version('numpy')}" in info
    assert f"* scipy version: {original_version('scipy')}" in info
