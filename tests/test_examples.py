import shlex


def test_2bin_1channel(tmp_path, script_runner):
    command = f"pyhf inspect {'docs/examples/json/2-bin_1-channel.json':s}"
    ret = script_runner.run(shlex.split(command))
    assert ret.success
