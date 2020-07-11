'''Shell completions for pyhf.'''
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
        '''Generate shell completion code for various shells.'''
        click.echo(click_completion.core.get_code(shell, prog_name='pyhf'))


except ImportError:

    @click.command(help='Generate shell completion code.', name='completions')
    @click.argument('shell', default=None)
    def cli(shell):
        '''Placeholder for shell completion code generatioon function if necessary dependency is missing.'''
        click.secho(
            'This requires the click_completion module.\n'
            'You can install it with the shellcomplete extra:\n'
            'python -m pip install pyhf[shellcomplete]'
        )
