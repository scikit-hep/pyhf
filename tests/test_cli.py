from click.testing import CliRunner


# FIXME
def test_shllcomplete_cli(isolate_modules):
    from pyhf.cli.complete import cli

    runner = CliRunner()
    result = runner.invoke(cli, ['bash'])
    assert 'complete -F _pyhf_completion -o default pyhf' in result.output
