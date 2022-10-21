import shutil
import sys
from pathlib import Path

import nox

ALL_PYTHONS = ["3.8", "3.9", "3.10"]

nox.options.sessions = ["lint", "tests"]


DIR = Path(__file__).parent.resolve()


@nox.session(reuse_venv=True)
def lint(session):
    """
    Lint with pre-commit.
    """
    session.install("--upgrade", "pre-commit")
    session.run("pre-commit", "run", "--all-files", *session.posargs)


@nox.session(python=ALL_PYTHONS, reuse_venv=True)
def tests(session):
    """
    Run the unit and regular tests.
    Specify a particular Python version with --python option.

    Example:

        $ nox --session tests --python 3.10
    """
    session.install("--upgrade", "--editable", ".[test]")

    # Allow for running a pytest-style keywords selection of the tests
    # c.f. https://docs.pytest.org/en/latest/how-to/usage.html#specifying-tests-selecting-tests
    if session.posargs and "-k" in session.posargs:
        session.posargs.pop(session.posargs.index("-k"))
        session.run("pytest", *session.posargs)
    else:
        session.run(
            "pytest",
            "--ignore",
            "tests/benchmarks/",
            "--ignore",
            "tests/contrib",
            "--ignore",
            "tests/test_notebooks.py",
            *session.posargs,
        )
        if sys.platform.startswith("linux"):
            session.run(
                "pytest",
                "tests/contrib",
                "--mpl",
                "--mpl-baseline-path",
                "tests/contrib/baseline" * session.posargs,
            )


@nox.session(reuse_venv=True)
def regenerate(session):
    """
    Regenerate Matplotlib images.
    """
    session.install("--upgrade", "--editable", ".[test]")
    if not sys.platform.startswith("linux"):
        session.error(
            "Must be run from Linux, images will be slightly different on macOS"
        )
    session.run(
        "pytest", "--mpl-generate-path=tests/contrib/baseline", *session.posargs
    )


@nox.session(reuse_venv=True)
def docs(session):
    """
    Build the docs.
    Pass "serve" to serve.
    Pass "clean" to delete the build tree prior to build.

    Example:

        $ nox --session docs -- serve
    """
    session.install("--upgrade", "--editable", ".[backends,contrib,docs]")

    build_path = DIR / "docs" / "_build"

    if session.posargs and "clean" in session.posargs:
        if build_path.exists():
            session.log(f"Removing build tree: {build_path}")
            shutil.rmtree(build_path)
        session.posargs.pop(session.posargs.index("clean"))

    session.chdir(build_path.parent)
    # https://www.sphinx-doc.org/en/master/man/sphinx-build.html
    session.run(
        "sphinx-build", "-M", "html", ".", build_path.name, "-W", "--keep-going"
    )
    session.log(
        f"rsync -r {build_path / 'html' / '_static'} {build_path / 'html' / 'docs'}"
    )
    shutil.copytree(
        build_path / "html" / "_static",
        build_path / "html" / "docs" / "_static",
        dirs_exist_ok=True,
    )
    session.log(
        f"rsync -r {build_path.parent.parent / 'src' / 'pyhf' / 'schemas'} {build_path / 'html'}"
    )
    shutil.copytree(
        build_path.parent.parent / "src" / "pyhf" / "schemas",
        build_path / "html" / "schemas",
        dirs_exist_ok=True,
    )
    session.log(f"Build finished. The HTML pages are in {build_path / 'html'}.")

    if session.posargs:
        if "serve" in session.posargs:
            print("Launching docs at http://localhost:8001/ - use Ctrl-C to quit")
            session.run(
                "python", "-m", "http.server", "8001", "-d", str(build_path / "html")
            )
        else:
            print("Unsupported argument to docs")


@nox.session(reuse_venv=True)
def notebooks(session: nox.Session):
    """
    Run the notebook tests.
    """
    session.install("--upgrade", "--editable", ".[test]")
    session.run(
        "pytest",
        "--override-ini",
        "filterwarnings=",
        "tests/test_notebooks.py",
        *session.posargs,
    )


@nox.session
def build(session):
    """
    Build an SDist and wheel.
    """

    # cleanup previous build and dist dirs
    build_path = DIR / "build"
    if build_path.exists():
        shutil.rmtree(build_path)
    dist_path = DIR / "dist"
    if dist_path.exists():
        shutil.rmtree(dist_path)

    session.install("build")
    session.run("python", "-m", "build")
