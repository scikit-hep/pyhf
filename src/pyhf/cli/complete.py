import click
try:
    import click_completion
    click_completion.init()

    @click.command(help='Generate shell completion code.', name='completions')
    @click.argument(
        'shell',
        required=False,
        type=click_completion.DocumentedChoice(click_completion.core.shells),
    )
    def cli(shell):
        click.echo(click_completion.core.get_code(shell))

except ImportError:
    @click.command(help='Generate shell completion code.', name='completions')
    @click.argument(
        'shell', default = None
    )
    def cli(shell):
        click.secho(
            'This requires the click_completion module.\n'
            'You can install it with the shellcomplete extra:\n'
            'python -m pip install pyhf[shellcomplete]'
        )