import shutil
import sys
from pathlib import Path

import nox

ALL_PYTHONS = ["3.8", "3.9", "3.10"]

# Default sessions to run if no session handles are passed
nox.options.sessions = ["lint", "tests-3.10"]


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

    Examples:

        $ nox --session tests --python 3.10
        $ nox --session tests --python 3.10 -- contrib  # run the contrib module tests
        $ nox --session tests --python 3.10 -- tests/test_tensor.py  # run specific tests
        $ nox --session tests --python 3.10 -- nocov  # run without coverage but faster
    """
    session.install("--upgrade", "--editable", ".[test]")
    session.install("--upgrade", "pytest", "coverage[toml]")

    # Allow tests to be run without coverage
    if "nocov" in session.posargs:
        runner_commands = ["pytest"]
        session.posargs.pop(session.posargs.index("nocov"))
    else:
        runner_commands = ["coverage", "run", "--append", "--module", "pytest"]

    def _contrib(session):
        if sys.platform.startswith("linux"):
            session.run(
                *runner_commands,
                "tests/contrib",
                "--mpl",
                "--mpl-baseline-path",
                "tests/contrib/baseline",
                "--mpl-generate-summary",
                "html",
                *session.posargs,
            )

    # Allow for running of contrib tests only
    if session.posargs and "contrib" in session.posargs:
        session.posargs.pop(session.posargs.index("contrib"))
        session.install("--upgrade", "matplotlib")
        _contrib(session)
        return

    if session.posargs:
        session.run(*runner_commands, *session.posargs)
    else:
        # defaults
        default_runner_commands = runner_commands.copy()
        if "--append" in default_runner_commands:
            default_runner_commands.pop(default_runner_commands.index("--append"))
        session.run(
            *default_runner_commands,
            "--ignore",
            "tests/contrib",
            "--ignore",
            "tests/benchmarks",
            "--ignore",
            "tests/test_notebooks.py",
        )
        _contrib(session)


@nox.session(reuse_venv=True)
def coverage(session):
    """
    Generate coverage report
    """
    session.install("--upgrade", "pip")
    session.install("--upgrade", "coverage[toml]")

    session.run("coverage", "report")
    session.run("coverage", "xml")
    htmlcov_path = DIR / "htmlcov"
    if htmlcov_path.exists():
        session.log(f"rm -r {htmlcov_path}")
        shutil.rmtree(htmlcov_path)
    session.run("coverage", "html")


@nox.session(reuse_venv=True)
def regenerate(session):
    """
    Regenerate Matplotlib images.
    """
    session.install("--upgrade", "--editable", ".[test]")
    session.install("--upgrade", "pytest", "matplotlib")
    if not sys.platform.startswith("linux"):
        session.error(
            "Must be run from Linux, images will be slightly different on macOS"
        )
    session.run(
        "pytest",
        "--mpl-generate-path=tests/contrib/baseline",
        "tests/contrib/test_viz.py",
        *session.posargs,
    )


@nox.session(reuse_venv=True)
def docs(session):
    """
    Build the docs.
    Pass "serve" to serve.
    Pass "clean" to delete the build tree prior to build.

    Example:

        $ nox --session docs -- serve
        $ nox --session docs -- clean
    """
    session.install("--upgrade", "--editable", ".[backends,contrib,docs]")
    session.install("--upgrade", "sphinx")

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
    Build a sdist and wheel.
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
