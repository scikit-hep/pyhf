"""Shell completions for pyhf."""

import click


@click.command(help="Generate shell completion code.", name="completions")
@click.argument(
    "shell",
    required=False,
    type=click.Choice(["bash", "zsh", "fish"], case_sensitive=False),
)
def cli(shell):
    """Generate shell completion code for various shells.

    Supported shells: bash, zsh, fish

    To enable completion, run the appropriate command for your shell:

    \b
    Bash:
      _PYHF_COMPLETE=bash_source pyhf > ~/.pyhf-complete.bash
      echo ". ~/.pyhf-complete.bash" >> ~/.bashrc

    \b
    Zsh:
      _PYHF_COMPLETE=zsh_source pyhf > ~/.pyhf-complete.zsh
      echo ". ~/.pyhf-complete.zsh" >> ~/.zshrc

    \b
    Fish:
      _PYHF_COMPLETE=fish_source pyhf >> ~/.config/fish/completions/pyhf.fish
    """
    if shell is None:
        click.echo(cli.get_help(click.Context(cli)))
        return

    click.echo(f"To enable {shell} completion for pyhf run:\n")

    if shell == "bash":
        click.echo("echo 'eval \"$(_PYHF_COMPLETE=bash_source pyhf)\"' >> ~/.bashrc\n")
    elif shell == "zsh":
        click.echo("echo 'eval \"$(_PYHF_COMPLETE=zsh_source pyhf)\"' >> ~/.zshrc\n")
    elif shell == "fish":
        click.echo(
            "echo '_PYHF_COMPLETE=fish_source pyhf | source' >> ~/.config/fish/completions/pyhf.fish\n"
        )
    click.echo("and then source your shell configuration or restart your shell.")
