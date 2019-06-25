import shlex


def test_2bin_singlechannel(tmpdir, script_runner):
    command = 'pyhf inspect {0:s}'.format('pyhf/examples/2-bin_single-channel.json')
    ret = script_runner.run(*shlex.split(command))
    assert ret.success
