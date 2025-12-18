"""Shell completions for pyhf."""

import click


@click.command(name="completions")
@click.argument(
    "shell",
    required=False,
    type=click.Choice(["bash", "zsh", "fish"], case_sensitive=False),
)
def cli(shell):
    """
    Generate shell completion code for various shells.

    The necessary commands to enable completions for pyhf for the specified
    shell will be printed to stdout.
    """
    if shell is None:
        click.echo(cli.get_help(click.Context(cli)))
        return

    click.echo(f"To enable {shell} completion for pyhf run in your {shell} shell:\n")

    instructions = {
        "bash": (
            "mkdir -p ~/.completions\n"
            "_PYHF_COMPLETE=bash_source pyhf > ~/.completions/pyhf-complete.sh\n"
            'echo -e "\\n. ~/.completions/pyhf-complete.sh" >> ~/.bashrc\n'
        ),
        "zsh": (
            "mkdir -p ~/.completions\n"
            "_PYHF_COMPLETE=zsh_source pyhf > ~/.completions/pyhf-complete.zsh\n"
            'echo -e "\\n. ~/.completions/pyhf-complete.zsh" >> ~/.zshrc\n'
        ),
        "fish": (
            "_PYHF_COMPLETE=fish_source pyhf >> ~/.config/fish/completions/pyhf.fish\n"
        ),
    }
    click.echo(
        click.style(
            instructions[shell],
            bold=True,
        )
    )
    click.echo(
        "and then source your shell configuration or restart your shell."
        + "\nPressing tab twice (<TAB><TAB>) will show all available subcommands."
        + "\nOptions are only listed if at least a dash has been entered (-<TAB><TAB>)."
    )
