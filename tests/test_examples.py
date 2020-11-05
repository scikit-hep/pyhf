import shlex


def test_2bin_1channel(tmpdir, script_runner):
    command = 'pyhf inspect {:s}'.format('docs/examples/json/2-bin_1-channel.json')
    ret = script_runner.run(*shlex.split(command))
    assert ret.success
