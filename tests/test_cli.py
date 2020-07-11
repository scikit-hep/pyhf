from click.testing import CliRunner
from pyhf.cli.cli import pyhf


def test_shllcomplete_cli():
    runner = CliRunner()
    result = runner.invoke(pyhf, ['completions', 'bash'])
    assert 'complete -F _pyhf_completion -o default pyhf' in result.output
