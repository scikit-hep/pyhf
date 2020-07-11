from click.testing import CliRunner
import sys
import importlib


def test_shllcomplete_cli(isolate_modules):
    from pyhf.cli.complete import cli

    runner = CliRunner()
    result = runner.invoke(cli, ['bash'])
    assert 'complete -F _pyhf_completion -o default pyhf' in result.output


def test_shllcomplete_cli_missing_extra(isolate_modules):
    sys.modules['click_completion'] = None
    importlib.reload(sys.modules['pyhf.cli.complete'])
    from pyhf.cli.complete import cli

    runner = CliRunner()
    result = runner.invoke(cli, ['bash'])
    assert 'You can install it with the shellcomplete extra' in result.output
