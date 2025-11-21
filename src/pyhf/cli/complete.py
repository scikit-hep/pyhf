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
      mkdir -p ~/.completions
      _PYHF_COMPLETE=bash_source pyhf > ~/.completions/pyhf-complete.sh
      echo ". ~/.completions/pyhf-complete.sh" >> ~/.bashrc

    \b
    Zsh:
      mkdir -p ~/.completions
      _PYHF_COMPLETE=zsh_source pyhf > ~/.completions/pyhf-complete.zsh
      echo ". ~/.completions/pyhf-complete.zsh" >> ~/.zshrc

    \b
    Fish:
      _PYHF_COMPLETE=fish_source pyhf >> ~/.config/fish/completions/pyhf.fish
    """
    if shell is None:
        click.echo(cli.get_help(click.Context(cli)))
        return

    click.echo(f"To enable {shell} completion for pyhf run in your {shell} shell:\n")

    if shell == "bash":
        click.echo(
            click.style(
                "mkdir -p ~/.completions\n"
                + "_PYHF_COMPLETE=bash_source pyhf > ~/.completions/pyhf-complete.sh\n"
                + 'echo -e "\\n. ~/.completions/pyhf-complete.sh" >> ~/.bashrc\n',
                bold=True,
            )
        )
    elif shell == "zsh":
        click.echo(
            click.style(
                "mkdir -p ~/.completions\n"
                + "_PYHF_COMPLETE=zsh_source pyhf > ~/.completions/pyhf-complete.zsh\n"
                + 'echo -e "\\n. ~/.completions/pyhf-complete.zsh" >> ~/.zshrc\n',
                bold=True,
            )
        )
    elif shell == "fish":
        click.echo(
            click.style(
                "_PYHF_COMPLETE=fish_source pyhf >> ~/.config/fish/completions/pyhf.fish\n",
                bold=True,
            )
        )
    click.echo("and then source your shell configuration or restart your shell.")
