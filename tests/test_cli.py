from click.testing import CliRunner


def test_shell_completion_cli_bash():
    from pyhf.cli.complete import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["bash"])
    assert result.exit_code == 0
    assert "_PYHF_COMPLETE=bash_source" in result.output
    assert ".bashrc" in result.output


def test_shell_completion_cli_zsh():
    from pyhf.cli.complete import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["zsh"])
    assert result.exit_code == 0
    assert "_PYHF_COMPLETE=zsh_source" in result.output
    assert ".zshrc" in result.output


def test_shell_completion_cli_fish():
    from pyhf.cli.complete import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["fish"])
    assert result.exit_code == 0
    assert "_PYHF_COMPLETE=fish_source" in result.output
    assert "fish/completions" in result.output


def test_shell_completion_cli_no_shell():
    from pyhf.cli.complete import cli

    runner = CliRunner()
    result = runner.invoke(cli)
    assert result.exit_code == 0
    assert "Generate shell completion code" in result.output
