from click.testing import CliRunner


# FIXME
# pyhf.cli.complete was removed given typer supports completion
# so need to rewrite and fix this
def test_shllcomplete_cli(isolate_modules):
    from pyhf.cli.complete import cli

    runner = CliRunner()
    result = runner.invoke(cli, ['bash'])
    assert 'complete -F _pyhf_completion -o default pyhf' in result.output
