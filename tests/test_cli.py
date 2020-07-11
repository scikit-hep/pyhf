from click.testing import CliRunner
import sys


def test_shllcomplete_cli():
    from pyhf.cli.cli import pyhf

    runner = CliRunner()
    result = runner.invoke(pyhf, ['completions', 'bash'])
    assert 'complete -F _pyhf_completion -o default pyhf' in result.output


def test_shllcomplete_cli_missing_extra(isolate_modules):
    import click_completion

    CACHE_MOD, sys.modules['click_completion'] = sys.modules['click_completion'], None
    from pyhf.cli.cli import pyhf

    runner = CliRunner()
    result = runner.invoke(pyhf, ['completions', 'bash'])
    assert 'You can install it with the shellcomplete extra' in result.output
    CACHE_MOD, sys.modules['click_completion'] = None, CACHE_MOD
